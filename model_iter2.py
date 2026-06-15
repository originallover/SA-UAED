import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, Wav2Vec2FeatureExtractor,AutoProcessor
from BEATs import BEATs, BEATsConfig
from convolution import Conv1dSubsampler
import torchaudio
import math
from Conv import ConvFeatureExtractionModel
from WavLM import WavLMConfig,WavLM

#预训练模型加载
class PretrainedModelLoader:
    @staticmethod
    def load_beats(freeze=True):
        "加载BEATs模型，使用BEATs_iter3+(AS2M)"
        checkpoint = torch.load('./BEATs_iter3_plus_AS2M.pt',weights_only=False)
        cfg = BEATsConfig(checkpoint['cfg'])
        BEATs_model = BEATs(cfg)
        BEATs_model.load_state_dict(checkpoint['model'])
        if freeze:
            for param in BEATs_model.parameters():
                param.requires_grad=False

        return BEATs_model

#瓶颈适配器
class BottleneckAdapter(nn.Module):
    def __init__(self, input_dim, bottleneck_dim, activation):
        super().__init__()
        self.down_proj = nn.Linear(input_dim, bottleneck_dim)
        self.non_linearity = F.relu if activation == 'relu' else F.gelu
        self.up_proj = nn.Linear(bottleneck_dim, input_dim)
        self.layer_norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        residual = x
        x = self.down_proj(x)
        x = self.non_linearity(x)
        x = self.up_proj(x)
        x = self.layer_norm(x)
        return x + residual  # 残差连接 
    
#声音适配器：时间对齐+瓶颈适配器+线性投影
class SoundAdapter(nn.Module):
    def __init__(self,input_dim=768,output_dim=192,bottleneck_dim=512):
        super().__init__()
        #1,时间对齐
        self.conv1d=Conv1dSubsampler(input_dim,512,1024,[4,4])
        #2,瓶颈适配器
        self.bottleneck=BottleneckAdapter(1024,bottleneck_dim,activation='gelu')
        #3,线性投影
        self.projection=nn.Linear(1024,output_dim)

    def forward(self,x):
        x=self.conv1d(x)
        x=self.bottleneck(x)
        x=self.projection(x)
        return x

#说话人适配器
class SpeakerAdapter(nn.Module):
    def __init__(self,input_dim=1024,output_dim=192,
                 num_layers=25,bottleneck_dim=512):
        super().__init__()
         #初始化每层的权重
        initial_weights = torch.ones(num_layers, requires_grad=True).float()
        self.output_weights = nn.Parameter(initial_weights)
        self.conv1d=Conv1dSubsampler(input_dim,512,1024,[2,1])
        self.bottleneck=BottleneckAdapter(1024,bottleneck_dim,activation='gelu')
        self.projection=nn.Linear(1024,output_dim)

    def forward(self,hidden_states):
        #hidden_states: list[(torch.size(50*t-1, batch_size, 1024),gradient) x 25] 25层
        norm_output_weights = F.softmax(self.output_weights, dim=0)
        outputs = [output for output, _ in hidden_states]  # List of [seq_len, batch, dim] X 25
         # 堆叠成 [num_layers, seq_len, batch, dim]
        stacked_outputs = torch.stack(outputs, dim=0)
        # 调整 weights 维度: [num_layers] -> [num_layers, 1, 1, 1]
        weights = norm_output_weights.view(-1, 1, 1, 1)
        # 加权求和: [seq_len, batch, dim]
        x = (stacked_outputs * weights).sum(dim=0)
        x=x.transpose(0,1)

        x=self.conv1d(x)
        x=self.bottleneck(x)
        x=self.projection(x)
        return x
    
#完整听觉编码器
class AuditoryEncoder(nn.Module):
    def __init__(self,output_dim=192,target_frame_rate=50):
        super().__init__()
        self.output_dim=output_dim
        self.target_frame_rate=target_frame_rate
        "加载预训练模型"
        print("加载预训练模型")
        self.beats_model=PretrainedModelLoader.load_beats(freeze=True)
        self.sound_adapter=SoundAdapter(input_dim=768,output_dim=output_dim,
                                        bottleneck_dim=512)
        self.speaker_adapter=SpeakerAdapter(input_dim=1024,output_dim=output_dim,
                                            num_layers=25,bottleneck_dim=512)
        #加载CNN提取模块和WavLM模型
        checkpoint = torch.load('./WavLM-Large.pt',weights_only=False )
        cfg = WavLMConfig(checkpoint['cfg'])
        feature_enc_layers = eval(cfg.conv_feature_layers)
        self.wavlm_model = WavLM(cfg)
        self.wavlm_model.load_state_dict(checkpoint['model'])
        for param in self.wavlm_model.parameters():
            param.requires_grad = False
        self.cnn_moudle=ConvFeatureExtractionModel(conv_layers=feature_enc_layers,
                                             dropout=0.0,
                                             mode=cfg.extractor_mode,
                                             conv_bias=cfg.conv_bias)
        self.linear=nn.Linear(512,output_dim)
        #特征融合层
        self.fusion_conv=nn.Conv1d(output_dim*3,output_dim,kernel_size=1)

        self.layer_norm=nn.LayerNorm(output_dim)
    
    def forward(self,waveform):
        batch_size=waveform[0]
        #=======BEATs特征提取==========
        
        beats_output=self.beats_model.extract_features(waveform)[0]
        beats_adapter=self.sound_adapter(beats_output)# B x T x C
        #=======WavLM特征提取==========
        
        rep, layer_results = self.wavlm_model.extract_features(waveform, output_layer=self.wavlm_model.cfg.encoder_layers, ret_layer_results=True)[0]
        wavlm_adapter=self.speaker_adapter(layer_results)# B x T x C
        #=======CNN特征提取==================
        
            #B x C(512) x T
        cnn_features=self.cnn_moudle(waveform)
        cnn_features=self.linear(cnn_features.transpose(1,2))# B x T x C(192)
            # print(cnn_features.shape)
        #=======特征融合==================
        target_num_frames=max(beats_adapter.shape[1],wavlm_adapter.shape[1],cnn_features.shape[1])
        beats_adapter=F.interpolate(beats_adapter.transpose(1,2),size=target_num_frames,mode='linear',align_corners=False).transpose(1,2)
        wavlm_adapter=F.interpolate(wavlm_adapter.transpose(1,2),size=target_num_frames,mode='linear',align_corners=False).transpose(1,2)
        cnn_features=F.interpolate(cnn_features.transpose(1,2),size=target_num_frames,mode='linear',align_corners=False).transpose(1,2)
        fused=torch.cat([beats_adapter,wavlm_adapter,cnn_features],dim=-1)
        fused=fused.transpose(1,2)
        fused=self.fusion_conv(fused)
        fused=fused.transpose(1,2)
        fused=self.layer_norm(fused)

        return fused  # B x T x C

#======================UEAD Quereies模块======================
class UAEDQueries(nn.Module):
    def __init__(self,num_sound_events=9,speaker_embedding_dim=192,query_dim=192):
        super().__init__()
        self.num_sound_events=num_sound_events
        self.query_dim=query_dim
        self.speaker_embedding_dim=speaker_embedding_dim
        #非语音事件的可学习查询
        self.sound_event_queries=nn.Parameter(torch.randn(self.num_sound_events,self.query_dim))
        #将说话人嵌入投影到query_dim
        self.speaker_projection=nn.Linear(speaker_embedding_dim,query_dim)
        # 将说话人嵌入映射到两个副语言嵌入空间
        self.speaker_to_cough = nn.Linear(speaker_embedding_dim, query_dim)
        self.speaker_to_laugh = nn.Linear(speaker_embedding_dim, query_dim)
    def forward(self,speaker_embeddings=None):
        """speaker_embedding:[batch,num_speaker,speaker_embedding_dim]
        Return:queries:[batch,num_sound_events+num_speakers,query_dim]"""
        batch_size = speaker_embeddings.size(0)
        #非语音事件查询
        sound_queries=self.sound_event_queries.unsqueeze(0).expand(batch_size,-1,-1)  # [batch,num_sound_events,query_dim]
        if speaker_embeddings is not None:
            #投影说话人嵌入到192维
            speaker_queries=self.speaker_projection(speaker_embeddings)
            cough_queries=self.speaker_to_cough(speaker_embeddings)
            laugh_queries=self.speaker_to_laugh(speaker_embeddings)
            #拼接所有查询
            queries=torch.cat([sound_queries,speaker_queries,cough_queries,laugh_queries],dim=1)
        else:
            queries=sound_queries
        return queries


#======================Transformer Encoder==========================
class TransformerEncoder(nn.Module):
    def __init__(self,d_model=192,nhead=8,num_layers=6,
                 dim_feedforward=768,dropout=0.1):
        super().__init__()
        # 移除位置编码
        # self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )

    def forward(self, src, src_mask=None):
        """
        src: [batch, seq_len, d_model]
        src_mask: padding mask (optional)
        output: [batch, seq_len, d_model]
        """
        # 直接使用输入，不添加位置编码
        output = self.transformer_encoder(src, src_key_padding_mask=src_mask)
        return output

#======================Transformer Decoder================================
class TransformerDecoder(nn.Module):
    def __init__(self, d_model=192, nhead=8, num_layers=6,
                 dim_feedforward=768, dropout=0.1):  # 修正了 droupout 拼写错误
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,  # 使用传入的参数而不是硬编码的768
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, 
            num_layers=num_layers
        )

    def forward(self, tgt, memory, memory_mask=None):
        """
        tgt: [batch, num_queries, d_model] -- 查询向量
        memory: [batch, seq_len, d_model] -- 编码器输出
        memory_mask: encoder的padding mask (optional)
        output: [batch, num_queries, d_model]
        
        移除了tgt_mask参数（因果掩码）
        """
        # 不传入tgt_mask，移除因果掩码机制
        output = self.transformer_decoder(
            tgt, 
            memory, 
            tgt_mask=None,  # 移除因果掩码
            memory_key_padding_mask=memory_mask
        )
        return output
#======================T-UAED模型================================
class TUAED(nn.Module):
    def __init__(self,num_sound_events=9,d_model=192,
                 nhead=8,num_encoder_layers=6,num_decoder_layers=6,
                 dim_feedforward=768,dropout=0.1):
        super().__init__()
        #1、听觉编码器
        self.auditory_encoder=AuditoryEncoder(output_dim=d_model)
        #2、UAED查询机制
        self.uaed_queries=UAEDQueries(num_sound_events=num_sound_events,
                                      speaker_embedding_dim=192,
                                      query_dim=d_model)
        #3、Transformer编码器
        self.encoder=TransformerEncoder(d_model=d_model,nhead=nhead,num_layers=num_encoder_layers,
                                        dim_feedforward=dim_feedforward,dropout=dropout)
        #4、Transformer解码器
        self.decoder=TransformerDecoder(d_model=d_model,nhead=nhead,num_layers=num_decoder_layers,
                                        dim_feedforward=dim_feedforward,dropout=dropout)
    
    def forward(self,audio,speaker_embeddings):

        batch_size=audio.size(0)
        #1、听觉编码器提取特征--FU:[batch,num_frames,d_model]
        FU=self.auditory_encoder(audio)
        num_frames=FU.size(1)

        #2、Transformer编码器--Fenc:[batch,num_frames,d_model]
        Fenc=self.encoder(FU)

        #3、生成UAED查询--queries:[batch,num_events,d_model]
        queries=self.uaed_queries(speaker_embeddings)
        #扩展到batch维度--queries:[batch,num_events,d_model]
        # queries=queries.unsqueeze(0).expand(batch_size,-1,-1)

        #4、Transformer解码器--Fdec:[batch,num_events,d_model]
        Fdec=self.decoder(queries,Fenc)

        #5、计算预测--predictions:[batch,num_events,num_frames]
        predictions=torch.bmm(Fdec,Fenc.transpose(1,2))
        predictions=torch.sigmoid(predictions)

        return predictions

#====================损失函数==========================
class UAEDLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce=nn.BCELoss(reduction='mean')

    def forward(self,predictions,targets):
        """predictions:[batch,num_events,num_frames]
        targets:[batch,num_events,num_frames]
        return loss"""
        return self.bce(predictions,targets)

if __name__=="__main__":
    #创建模拟数据
    batch_size=4
    sample_rate=16000
    duration=10
    waveform=torch.randn(batch_size,sample_rate*duration)
    speaker_embeddings=torch.randn(4,3,192)

    model = TUAED()
    predictions=model(waveform,speaker_embeddings)
    print(predictions.shape)  # 应输出: [4, num_sound_events + num_speakers, num_frames]

    targets=torch.randint(0,2,predictions.shape).float()
    criterion=UAEDLoss()
    loss=criterion(predictions,targets)
    print(f"Loss: {loss.item()}")