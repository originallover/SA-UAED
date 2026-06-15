#!/usr/bin/env python3
"""
合成对话音频
从 session 级别音频中提取片段并拼接
"""

import numpy as np
import soundfile as sf
import os
import argparse
from pathlib import Path
from typing import Dict, List
import subprocess
from multiprocessing import Pool
from tqdm import tqdm
import json
import librosa


class ConversationSynthesizer:
    """对话音频合成器"""
    
    def __init__(self, wav_scp: Dict[str, str], sampling_rate: int = 16000):
        self.wav_scp = wav_scp
        self.sampling_rate = sampling_rate
    
    def load_audio_segment(self, wav_path: str, start_frame: int,
                          duration_frames: int) -> np.ndarray:
        """
        从音频文件加载片段
        """
        print(f"[DEBUG] Loading: {wav_path[:100]}...")
        # print(f"[DEBUG]   start={start_frame}, duration={duration_frames}")
        # 解析 wav.scp 格式
        if '|' in wav_path:
            parts = wav_path.split()
            for i, part in enumerate(parts):
                if part == '-s' and i+1 < len(parts):
                    file_path = parts[i+1].rstrip('|').strip()  # 提取文件路径
                    break
        else:
            raise ValueError(f"Invalid wav_path format: {wav_path}")
        
        # 直接读取文件
        audio, sr = sf.read(file_path)
        if sr != self.sampling_rate:
            raise ValueError(f"Sampling rate mismatch: {sr} vs {self.sampling_rate}")
        print(f"[DEBUG] Loaded audio: {len(audio)} samples")
        # 提取片段
        end_frame = start_frame + duration_frames
        if end_frame > len(audio):
            # 如果超出范围，循环填充
            audio = np.tile(audio, int(np.ceil(end_frame / len(audio))))
        segment = audio[start_frame:end_frame]
        
        return segment
    
    #加载事件音频
    def load_audio_event(self, wav_path: str, 
                          duration: int) -> np.ndarray:
        audio, sr = librosa.load(wav_path, sr=self.sampling_rate,mono=True)
        total_frames = int(duration * self.sampling_rate)
        event_audio = audio[0:total_frames]
        return event_audio

    #按SNR混合音频
    def mix_audio_with_snr(self,signal_audio, noise_audio, snr_db):
        # 计算信号和噪声的功率
        signal_power = np.mean(signal_audio**2)
        noise_power = np.mean(noise_audio**2)
    #计算缩放系数   
        if snr_db == 0:
            alpha = np.sqrt(signal_power / noise_power)
            scaled_noise = noise_audio * alpha
            return scaled_noise
        else:
            alpha = np.sqrt(signal_power /(10**(snr_db / 10)) / noise_power)
            scaled_noise = noise_audio * alpha
            return scaled_noise
    #添加白噪声
    def add_white_noise(self,audio):
        signal_power = np.mean(audio**2)
        if signal_power < 1e-10:
            signal_power = 1e-6
        
        snr_db = np.random.uniform(20,40)
        noise = np.random.normal(0, 1, len(audio))
        noise_power = np.mean(noise**2)
        alpha = np.sqrt(signal_power /(10**(snr_db / 10)) / noise_power)
        scaled_noise = noise * alpha
        noisy_audio = audio + scaled_noise
        return noisy_audio

    def synthesize_conversation(self, conv_file: str) -> np.ndarray:
        """
        合成一个对话
        
        Args:
            conv_file: .conv 文件路径
            
        Returns:
            conversation: 合成的对话音频
        """
        # 读取 .conv 文件
        segments = []
        with open(conv_file, 'r') as f:
            conv = json.load(f)
            segments = conv['segments']
            bg_events = conv.get('background_events', [])
            fg_events = conv.get('foreground_events', [])
            para_events = conv.get('paralinguistic_events', [])
       
        # 计算总长度
        max_end = 32 * self.sampling_rate  # 默认32秒
        conversation = np.zeros(max_end, dtype=np.float32)
        
        # 加载并混合每个片段
        for seg in segments:
            utt_id = seg['utterance_id']
            if utt_id not in self.wav_scp:
                print(f"Warning: {utt_id} not found in wav.scp")
                continue
            
            wav_path = self.wav_scp[utt_id]
            
            # 加载音频片段
            audio_segment = self.load_audio_segment(
                wav_path,
                seg['position_frames'],
                seg['duration_frames']
            )
            
            # 混合到对话中
            pos = seg['start_frame']
            end_pos = pos + len(audio_segment)
            
            # 处理重叠（简单相加）
            conversation[pos:end_pos] += audio_segment

        #加载前景事件
        fg_audio = []
        for event in fg_events:  
            wav_path = event['wav_path']
            
            # 加载音频片段
            audio_event = self.load_audio_event(
                wav_path,
                event['duration']
            )
            snr_db = np.random.uniform(0,5)
            scaled_audio_event = self.mix_audio_with_snr(conversation, audio_event, snr_db)
            fg_audio.append(scaled_audio_event)
        
        # 加载背景事件
        bg_audio = []
        for event in bg_events:
            wav_path = event['wav_path']
            audio_event = self.load_audio_event(
                wav_path,
                event['duration']
            )
            snr_db = np.random.uniform(10,15)
            scaled_audio_event = self.mix_audio_with_snr(conversation, audio_event, snr_db)
            bg_audio.append(scaled_audio_event)

        # 混合前景事件
        for i in range(len(fg_audio)):
            start_frame = int(fg_events[i]['start_time'] * self.sampling_rate)
            end_frame = start_frame + len(fg_audio[i])
            conversation[start_frame:end_frame] += fg_audio[i]
        # 混合背景事件
        for i in range(len(bg_audio)):
            start_frame = int(bg_events[i]['start_time'] * self.sampling_rate)
            end_frame = start_frame + len(bg_audio[i])
            conversation[start_frame:end_frame] += bg_audio[i]

        # 加载并混合paralinguistic事件（laugh/cough）
        para_audio = []
        for event in para_events:
            wav_path = event['wav_path']
            try:
                audio_event = self.load_audio_event(
                    wav_path,
                    event['duration']
                )
                # laugh和cough使用较高的SNR（更清晰），因为它们来自说话人
                snr_db = np.random.uniform(3, 8)
                scaled_audio_event = self.mix_audio_with_snr(conversation, audio_event, snr_db)
                para_audio.append(scaled_audio_event)
            except Exception as e:
                print(f"Warning: Failed to load paralinguistic audio {wav_path}: {e}")
                para_audio.append(None)
        
        # 混合paralinguistic事件
        for i in range(len(para_audio)):
            if para_audio[i] is not None:
                start_frame = int(para_events[i]['start_time'] * self.sampling_rate)
                end_frame = start_frame + len(para_audio[i])
                # 确保不超出边界
                if end_frame <= len(conversation):
                    conversation[start_frame:end_frame] += para_audio[i]
                else:
                    # 截断到边界
                    valid_len = len(conversation) - start_frame
                    if valid_len > 0:
                        conversation[start_frame:] += para_audio[i][:valid_len]

        # 归一化防止削波
        max_val = np.abs(conversation).max()
        print(f"max_val:{max_val}")
        if max_val > 0.99:
            conversation = conversation * 0.99 / max_val
        #添加白噪声
        conversation = self.add_white_noise(conversation)
        return conversation


def process_conversation(args):
    """处理单个对话（用于并行）"""
    conv_id, conv_dir, wav_scp, output_dir, sampling_rate = args
    
    try:
        synthesizer = ConversationSynthesizer(wav_scp, sampling_rate)
        
        conv_file = os.path.join(conv_dir, f'{conv_id}.json')
        conversation = synthesizer.synthesize_conversation(conv_file)
        
        output_file = os.path.join(output_dir, f'{conv_id}.wav')
        sf.write(output_file, conversation, sampling_rate)
        
        return True, conv_id
    except Exception as e:
        return False, f"{conv_id}: {str(e)}"


def load_wav_scp(wav_scp_file: str) -> Dict[str, str]:
    """加载 wav.scp"""
    wav_scp = {}
    with open(wav_scp_file, 'r') as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                wav_scp[parts[0]] = parts[1]
    return wav_scp


def main():
    parser = argparse.ArgumentParser(
        description='Synthesize conversation audio'
    )
    parser.add_argument('--conv-list', required=True,
                       help='Conversation list file')
    parser.add_argument('--conv-dir', required=True,
                       help='Directory with .conv files')
    parser.add_argument('--wav-scp', required=True,
                       help='wav.scp file')
    parser.add_argument('--output-dir', required=True,
                       help='Output directory for audio')
    parser.add_argument('--sampling-rate', type=int, default=16000)
    parser.add_argument('--num-workers', type=int, default=8,
                       help='Number of parallel workers')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载 wav.scp
    print(f"Loading wav.scp from {args.wav_scp}...")
    wav_scp = load_wav_scp(args.wav_scp)
    print(f"Loaded {len(wav_scp)} utterances")
    
    # 读取对话列表
    with open(args.conv_list, 'r') as f:
        conv_ids = [line.strip() for line in f]
    
    print(f"\nSynthesizing {len(conv_ids)} conversations...")
    print(f"Using {args.num_workers} workers")
    
    # 准备参数
    task_args = [
        (conv_id, args.conv_dir, wav_scp, args.output_dir, args.sampling_rate)
        for conv_id in conv_ids
    ]
    
    # 并行处理
    with Pool(args.num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_conversation, task_args),
            total=len(conv_ids)
        ))
    
    # 统计结果
    success_count = sum([1 for success, _ in results if success])
    failed = [msg for success, msg in results if not success]
    
    print(f"\nSynthesis complete!")
    print(f"  Success: {success_count}/{len(conv_ids)}")
    
    if failed:
        print(f"  Failed: {len(failed)}")
        print("  First 5 failures:")
        for msg in failed[:5]:
            print(f"    {msg}")


if __name__ == '__main__':
    main()
