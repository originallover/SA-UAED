#!/usr/bin/env python3
"""
生成对话定义文件
"""
import librosa
import numpy as np
import os
import argparse
from pathlib import Path
from typing import Dict, List
import json
from json import JSONEncoder
import glob

BACKGROUND_EVENTS = ["Blender","Electric_shaver_toothbrush","Frying","Running_water","Vacuum_cleaner"]
FOREGROUND_EVENTS = ["Alarm_bell_ringing","Blender","Cat","Dishes","Dog","Electric_shaver_toothbrush","Frying","Running_water","Vacuum_cleaner"]
PARALINGUISTIC_EVENTS = ["laugh", "cough"]

    
def background_generator(events) -> List[Dict]:
    """背景事件生成器（占位符）"""
    bk_list=[]
    if np.random.rand() < 0.5:
        return bk_list
    else:
        event_class = np.random.choice(BACKGROUND_EVENTS)
        duration = np.random.uniform(10.0,30.0)
        wav_paths=list(events["background"][event_class])
        wav_path = np.random.choice(wav_paths)
        duration_seconds = librosa.get_duration(path=wav_path)
        duration = min(duration,duration_seconds)
        start_time = np.random.uniform(0,32.0 - duration)
        event={
            "event_class":event_class,
            "wav_path":wav_path,
            "duration":duration,
            "start_time":start_time
        }
        bk_list.append(event)
        return bk_list

def foreground_generator(events) -> List[Dict]:
    """前景事件生成器（占位符）"""
    fg_list=[]
    num_events = np.random.randint(1,3)
    event_classes = np.random.choice(FOREGROUND_EVENTS,num_events)
    i = 0
    while(i<num_events):
        event_class = event_classes[i]
        duration = np.random.uniform(1.0,5.0)
        wav_paths=list(events["foreground"][event_class])
        wav_path = np.random.choice(wav_paths)
        duration_seconds = librosa.get_duration(path=wav_path)
        duration = min(duration,duration_seconds)
        start_time = np.random.uniform(0,32.0 - duration)
        event={
            "event_class":event_class,
            "wav_path":wav_path,
            "duration":duration,
            "start_time":start_time
        }
        fg_list.append(event)
        i+=1
    return fg_list


def load_paralinguistic_index(paralinguistic_dir: str) -> Dict[str, Dict[str, List[str]]]:
    """
    加载paralinguistic（laugh/cough）文件索引
    
    目录结构: paralinguistic_dir/{train-clean-100, train-clean-360, ...}/{speaker_id}/{laugh,cough}/*.wav
    
    Returns:
        Dict: {speaker_id: {"laugh": [wav_paths], "cough": [wav_paths]}}
    """
    paralinguistic_index = {}
    
    if not os.path.exists(paralinguistic_dir):
        print(f"Warning: Paralinguistic directory not found: {paralinguistic_dir}")
        return paralinguistic_index
    
    # 遍历所有子集目录
    subsets = ["train-clean-100", "train-clean-360", "train-other-500", "dev-clean"]
    for subset in subsets:
        subset_dir = os.path.join(paralinguistic_dir, subset)
        if not os.path.exists(subset_dir):
            continue
        
        # 遍历每个speaker目录
        for speaker_id in os.listdir(subset_dir):
            speaker_path = os.path.join(subset_dir, speaker_id)
            if not os.path.isdir(speaker_path):
                continue
            
            if speaker_id not in paralinguistic_index:
                paralinguistic_index[speaker_id] = {"laugh": [], "cough": []}
            
            # 加载laugh文件
            laugh_dir = os.path.join(speaker_path, "laugh")
            if os.path.exists(laugh_dir):
                laugh_files = glob.glob(os.path.join(laugh_dir, "*.wav"))
                paralinguistic_index[speaker_id]["laugh"].extend(laugh_files)
            
            # 加载cough文件
            cough_dir = os.path.join(speaker_path, "cough")
            if os.path.exists(cough_dir):
                cough_files = glob.glob(os.path.join(cough_dir, "*.wav"))
                paralinguistic_index[speaker_id]["cough"].extend(cough_files)
    
    # 统计信息
    total_speakers = len(paralinguistic_index)
    total_laugh = sum(len(v["laugh"]) for v in paralinguistic_index.values())
    total_cough = sum(len(v["cough"]) for v in paralinguistic_index.values())
    print(f"Loaded paralinguistic index: {total_speakers} speakers, {total_laugh} laugh files, {total_cough} cough files")
    
    return paralinguistic_index


def paralinguistic_generator(timeline: List[Dict], paralinguistic_index: Dict, 
                             total_duration: float = 32.0,
                             laugh_prob: float = 0.3, cough_prob: float = 0.2) -> List[Dict]:
    """
    为对话中的说话人生成laugh和cough事件
    
    Args:
        timeline: 已分配好utterance的时间轴
        paralinguistic_index: {speaker_id: {"laugh": [wav_paths], "cough": [wav_paths]}}
        total_duration: 对话总时长
        laugh_prob: 每个segment添加laugh的概率
        cough_prob: 每个segment添加cough的概率
    
    Returns:
        List[Dict]: paralinguistic事件列表
    """
    para_events = []
    
    if not paralinguistic_index:
        return para_events
    
    for seg in timeline:
        real_speaker_id = str(seg.get('real_speaker_id', ''))
        
        # 检查该speaker是否有paralinguistic文件
        if real_speaker_id not in paralinguistic_index:
            continue
        
        speaker_para = paralinguistic_index[real_speaker_id]
        
        # 随机决定是否添加laugh
        if speaker_para["laugh"] and np.random.rand() < laugh_prob:
            wav_path = np.random.choice(speaker_para["laugh"])
            try:
                duration_seconds = librosa.get_duration(path=wav_path)
                # laugh通常较短，限制在0.5-3秒
                duration = min(duration_seconds, np.random.uniform(0.5, 2.0))
                
                # 在segment前后随机位置插入
                # 可以在segment开始前、结束后、或者segment中间
                insert_position = np.random.choice(['before', 'after', 'during'])
                
                if insert_position == 'before':
                    start_time = max(0, seg['start_time'] - duration - np.random.uniform(0, 0.3))
                elif insert_position == 'after':
                    start_time = min(total_duration - duration, seg['end_time'] + np.random.uniform(0, 0.3))
                else:  # during
                    # 在segment中间随机位置
                    max_start = seg['end_time'] - duration
                    if max_start > seg['start_time']:
                        start_time = np.random.uniform(seg['start_time'], max_start)
                    else:
                        start_time = seg['start_time']
                
                # 确保不超出边界
                start_time = max(0, min(start_time, total_duration - duration))
                
                event = {
                    "event_class": "laugh",
                    "wav_path": wav_path,
                    "duration": duration,
                    "start_time": start_time,
                    "speaker_id": real_speaker_id
                }
                para_events.append(event)
            except Exception as e:
                print(f"Warning: Failed to process laugh file {wav_path}: {e}")
        
        # 随机决定是否添加cough
        if speaker_para["cough"] and np.random.rand() < cough_prob:
            wav_path = np.random.choice(speaker_para["cough"])
            try:
                duration_seconds = librosa.get_duration(path=wav_path)
                # cough通常较短
                duration = min(duration_seconds, np.random.uniform(0.3, 1.5))
                
                # cough更可能出现在说话前后
                insert_position = np.random.choice(['before', 'after', 'during'], p=[0.4, 0.4, 0.2])
                
                if insert_position == 'before':
                    start_time = max(0, seg['start_time'] - duration - np.random.uniform(0, 0.2))
                elif insert_position == 'after':
                    start_time = min(total_duration - duration, seg['end_time'] + np.random.uniform(0, 0.2))
                else:  # during
                    max_start = seg['end_time'] - duration
                    if max_start > seg['start_time']:
                        start_time = np.random.uniform(seg['start_time'], max_start)
                    else:
                        start_time = seg['start_time']
                
                # 确保不超出边界
                start_time = max(0, min(start_time, total_duration - duration))
                
                event = {
                    "event_class": "cough",
                    "wav_path": wav_path,
                    "duration": duration,
                    "start_time": start_time,
                    "speaker_id": real_speaker_id
                }
                para_events.append(event)
            except Exception as e:
                print(f"Warning: Failed to process cough file {wav_path}: {e}")
    
    return para_events


class ConversationGenerator:
    """
    对话生成器
    使用统计参数
    """
    def __init__(self, sampling_rate: int = 16000, seed: int = 42):
        self.sampling_rate = sampling_rate
        np.random.seed(seed)
        
        self.utt_duration_mean = 5.0      # 平均发言时长（秒）
        self.utt_duration_std = 1.5
        self.utt_duration_min = 3.0
        self.utt_duration_max = 7.0
        
        self.silence_mean = 0.5           # 平均静音时长（秒）
        self.silence_std = 0.4
        self.silence_min = 0.1
        self.silence_max = 2.5
        
        self.overlap_probability = 0.15   # 15% 重叠概率30%
        self.overlap_min = 0.2
        self.overlap_max = 1.0
        
        self.same_speaker_prob = 0.1      # 同一说话人连续概率
        self.speaker_weights = [0.45, 0.35, 0.20]  # 说话人活跃度
    
    def _sample_duration(self, mean, std, min_val, max_val):
        """采样时长（截断正态分布）"""
        while True:
            duration = np.random.normal(mean, std)
            if min_val <= duration <= max_val:
                return duration
    
    def _select_speaker(self, current_speaker, num_speakers):
        """选择下一个说话人"""
        if current_speaker is None:
            weights = np.array(self.speaker_weights[:num_speakers])
            weights = weights / weights.sum()
            return np.random.choice(num_speakers, p=weights)
        
        if np.random.random() < self.same_speaker_prob:
            return current_speaker
        
        others = [i for i in range(num_speakers) if i != current_speaker]
        weights = np.array([self.speaker_weights[i] for i in others])
        weights = weights / weights.sum()
        return np.random.choice(others, p=weights)
    
    def generate_timeline(self, duration: float, num_speakers: int) -> List[Dict]:
        """
        生成对话时间轴
        
        Returns:
            timeline: 对话片段列表
        """
        timeline = []
        current_time = 0.0
        segment_id = 0
        current_speaker = None
        
        while current_time < duration:
            # 选择说话人
            speaker_id = self._select_speaker(current_speaker, num_speakers)
            
            # 采样发言时长
            utt_duration = self._sample_duration(
                self.utt_duration_mean, self.utt_duration_std,
                self.utt_duration_min, self.utt_duration_max
            )
            
            # 确保不超过总时长
            remaining = duration - current_time
            if utt_duration > remaining:
                utt_duration = remaining
            
            if utt_duration < 0.3:
                break
            
            # 决定是否重叠
            overlap = False
            overlap_offset = 0.0
            if current_time > 0 and np.random.random() < self.overlap_probability:
                overlap = True
                overlap_offset = np.random.uniform(self.overlap_min, self.overlap_max)
                overlap_offset = min(overlap_offset, current_time)
            
            # 计算帧位置
            actual_start = current_time - overlap_offset
            start_frame = int(actual_start * self.sampling_rate)
            duration_frames = int(utt_duration * self.sampling_rate)
            
            # 创建片段
            segment = {
                'segment_id': f'seg{segment_id:04d}',
                'speaker_id': speaker_id,
                'start_time': actual_start,
                'end_time': actual_start + utt_duration,
                'duration': utt_duration,
                'start_frame': start_frame,
                'duration_frames': duration_frames,
                'overlap': overlap
            }
            
            timeline.append(segment)
            
            # 采样静音间隔
            silence = self._sample_duration(
                self.silence_mean, self.silence_std,
                self.silence_min, self.silence_max
            )
            
            current_time += utt_duration + silence 
            current_speaker = speaker_id
            segment_id += 1
        
        return timeline
    
    def assign_utterances(self, timeline: List[Dict],
                         utt2spk: Dict, wav_scp: Dict) -> List[Dict]:
        """为时间轴分配真实 utterances"""
        
        # 按说话人分组
        spk2utts = {}
        for utt_id, spk_id in utt2spk.items():
            if utt_id in wav_scp:
                if spk_id not in spk2utts:
                    spk2utts[spk_id] = []
                spk2utts[spk_id].append(utt_id)
        
        all_speakers = list(spk2utts.keys())
        #算出的虚拟说话人数量可能大于真实的数量，当真实只有speaker0\speaker2时，会算出num_virtual_speakers=3
        num_virtual_speakers = max([seg['speaker_id'] for seg in timeline]) + 1
        
        # 为虚拟说话人分配真实说话人
        selected_speakers = np.random.choice(
            all_speakers,
            size=num_virtual_speakers,
            replace=False
        )
        
        # 分配 utterances
        timeline_with_files = []
        for seg in timeline:
            virtual_spk = seg['speaker_id']
            real_spk = selected_speakers[virtual_spk]
            available_utts = spk2utts[real_spk]
            
            selected_utt = np.random.choice(available_utts)
            
            seg_with_file = seg.copy()
            seg_with_file.update({
                'utterance_id': selected_utt,
                'wav_path': wav_scp[selected_utt],
                'real_speaker_id': real_spk
            })
            
            timeline_with_files.append(seg_with_file)
        
        return timeline_with_files
    
    # def save_conv_file(self, timeline: List[Dict], output_file: str):
    #     """保存为 .conv 格式"""
    #     with open(output_file, 'w') as f:
    #         dic={}
    #         for seg in timeline:
    #             if seg['utterance_id'] in dic:
    #                 line = (
    #                     f"{seg['segment_id']} "
    #                     f"{seg['utterance_id']} "
    #                     f"{dic[seg['utterance_id']]} "
    #                     f"{seg['duration_frames']} "
    #                     f"{seg['start_frame']}\n"
    #                 )
    #                 dic[seg['utterance_id']]+=seg['duration_frames']
    #             else:
    #                 dic[seg['utterance_id']]=seg['duration_frames']
    #                 line = (
    #                     f"{seg['segment_id']} "
    #                     f"{seg['utterance_id']} "
    #                     f"0 "
    #                     f"{seg['duration_frames']} "
    #                     f"{seg['start_frame']}\n"
    #                 )
    #             f.write(line)
    
    def save_metadata(self, timeline: List[Dict], bg_events, fg_events, 
                      para_events: List[Dict] = None, output_file: str = None):
        """保存元数据（JSON）"""
        dic = {}
        #当同一个人的同一段音频被使用时，避免重复
        for seg in timeline:
            if seg['utterance_id'] in dic:
                seg['position_frames'] = dic[seg['utterance_id']]
                dic[seg['utterance_id']] += seg['duration_frames']
            else:
                dic[seg['utterance_id']] = seg['duration_frames']
                seg['position_frames'] = 0
        metadata = {
            'num_segments': len(timeline),
            'total_duration': max([seg['end_time'] for seg in timeline]),
            'num_speakers': len(set([seg['speaker_id'] for seg in timeline])),
            'overlap_rate': sum([seg['overlap'] for seg in timeline]) / len(timeline),
            'segments': timeline,
            'background_events': bg_events,
            'foreground_events': fg_events,
            'paralinguistic_events': para_events if para_events else []
        }
        
        with open(output_file, 'w') as f:
            json.dump(metadata, f, indent=2, cls=NumpyEncoder)

class NumpyEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def load_kaldi_data(data_dir: str):
    """加载 Kaldi 数据"""
    utt2spk = {}
    wav_scp = {}
    
    with open(os.path.join(data_dir, 'utt2spk'), 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                utt2spk[parts[0]] = parts[1]
    
    with open(os.path.join(data_dir, 'wav.scp'), 'r') as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                wav_scp[parts[0]] = parts[1]
    
    return utt2spk, wav_scp


def main():
    parser = argparse.ArgumentParser(
        description='Generate conversation definition files'
    )
    parser.add_argument('--kaldi-data-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--sound-events-dir', required=True)
    parser.add_argument('--paralinguistic-dir', type=str, default=None,
                       help='Directory containing paralinguistic (laugh/cough) files')
    parser.add_argument('--num-conversations', type=int, required=True)
    parser.add_argument('--duration', type=float, default=32.0)
    parser.add_argument('--num-speakers', type=int, default=3)
    parser.add_argument('--sampling-rate', type=int, default=16000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save-metadata', action='store_true')
    parser.add_argument('--laugh-prob', type=float, default=0.3,
                       help='Probability of adding laugh event per segment')
    parser.add_argument('--cough-prob', type=float, default=0.2,
                       help='Probability of adding cough event per segment')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载数据
    print(f"Loading Kaldi data from {args.kaldi_data_dir}...")
    utt2spk, wav_scp = load_kaldi_data(args.kaldi_data_dir)
    print(f"Loaded {len(utt2spk)} utterances from {len(set(utt2spk.values()))} speakers")

    # 加载声音事件索引
    events_path = args.sound_events_dir
    with open(events_path, 'r') as f:
        events = json.load(f)
    
    # 加载paralinguistic索引（laugh/cough）
    paralinguistic_index = {}
    if args.paralinguistic_dir:
        print(f"\nLoading paralinguistic data from {args.paralinguistic_dir}...")
        paralinguistic_index = load_paralinguistic_index(args.paralinguistic_dir)
    
    # 初始化生成器
    generator = ConversationGenerator(args.sampling_rate, args.seed)
    
    # 生成对话
    print(f"\nGenerating {args.num_conversations} conversations...")
    conv_list = []
    
    for i in range(args.num_conversations):
        conv_id = f'conv_{i:06d}'
        
        # 生成时间轴
        timeline = generator.generate_timeline(args.duration, args.num_speakers)
        
        # 分配 utterances
        timeline_with_files = generator.assign_utterances(timeline, utt2spk, wav_scp)
        
        # 分配背景事件
        bg_events = background_generator(events)
        # 分配前景事件
        fg_events = foreground_generator(events)
        # 分配paralinguistic事件（laugh/cough）
        para_events = paralinguistic_generator(
            timeline_with_files, 
            paralinguistic_index,
            total_duration=args.duration,
            laugh_prob=args.laugh_prob,
            cough_prob=args.cough_prob
        )
        
        # 保存元数据
        meta_file = os.path.join(args.output_dir, f'{conv_id}.json')
        generator.save_metadata(timeline_with_files, bg_events, fg_events, 
                               para_events, meta_file)
        
        conv_list.append(conv_id)
        
        if (i + 1) % 1000 == 0:
            print(f"  Progress: {i + 1}/{args.num_conversations}")
    
    # 保存对话列表
    list_file = os.path.join(args.output_dir, 'conversations.list')
    with open(list_file, 'w') as f:
        for conv_id in conv_list:
            f.write(f"{conv_id}\n")
    
    print(f"\nDone! Conversation list saved to {list_file}")


if __name__ == '__main__':
    main()
