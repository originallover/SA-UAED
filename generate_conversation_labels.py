#!/usr/bin/env python3
"""
生成对话标签
为说话人分离任务生成帧级别标签
"""

import numpy as np
import os
import argparse
from pathlib import Path
import json
from typing import Dict, List
from json import JSONEncoder

EVENTS = ["Alarm_bell_ringing","Blender","Cat","Dishes","Dog","Electric_shaver_toothbrush","Frying","Running_water","Vacuum_cleaner"]
PARALINGUISTIC_EVENTS = ["laugh", "cough"]

class NumpyEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def generate_speaker_labels(metadata: Dict,
                           duration: float,
                           frame_rate: int = 50) -> Dict[str, np.ndarray]:
    """
    生成说话人标签
    """
    labels = {}
    num_frames = int(duration * frame_rate)
    
    # 初始化声音事件标签
    for event in EVENTS:
        labels[event] = np.zeros(num_frames, dtype=int)
    
    bg_events = metadata['background_events']
    fg_events = metadata['foreground_events']
    para_events = metadata.get('paralinguistic_events', [])
    
    # 填充背景事件标签
    for event in bg_events:
        start_frame = int(event['start_time'] * frame_rate)
        end_frame = start_frame + int(event['duration'] * frame_rate)
        end_frame = min(end_frame, num_frames)  # 防止越界
        labels[event['event_class']][start_frame:end_frame] = 1
    
    # 填充前景事件标签
    for event in fg_events:
        start_frame = int(event['start_time'] * frame_rate)
        end_frame = start_frame + int(event['duration'] * frame_rate)
        end_frame = min(end_frame, num_frames)  # 防止越界
        labels[event['event_class']][start_frame:end_frame] = 1

    segments = metadata['segments']
    num_speakers = metadata['num_speakers']

    # 填充说话人标签
    for seg in segments:
        real_speaker_id = seg['real_speaker_id']
        start_frame = int(seg['start_time'] * frame_rate)
        end_frame = start_frame + int(seg['duration'] * frame_rate)
        end_frame = min(end_frame, num_frames)  # 防止越界
        if f'speaker_{real_speaker_id}' not in labels:
            labels[f'speaker_{real_speaker_id}'] = np.zeros(num_frames, dtype=int)
        labels[f'speaker_{real_speaker_id}'][start_frame:end_frame] = 1
    
    # 填充paralinguistic事件标签（laugh/cough）
    # 为每个speaker单独创建laugh和cough标签
    for event in para_events:
        event_class = event['event_class']  # 'laugh' or 'cough'
        speaker_id = event.get('speaker_id', None)
        start_frame = int(event['start_time'] * frame_rate)
        end_frame = start_frame + int(event['duration'] * frame_rate)
        end_frame = min(end_frame, num_frames)  # 防止越界
        
        if speaker_id is not None:
            # 为每个speaker创建独立的laugh/cough标签
            label_key = f'{event_class}_{speaker_id}'
            if label_key not in labels:
                labels[label_key] = np.zeros(num_frames, dtype=int)
            labels[label_key][start_frame:end_frame] = 1
        
        # 同时也创建全局的laugh/cough标签（不区分speaker）
        # global_label_key = event_class
        # if global_label_key not in labels:
        #     labels[global_label_key] = np.zeros(num_frames, dtype=int)
        # labels[global_label_key][start_frame:end_frame] = 1
    
    return labels


def main():
    parser = argparse.ArgumentParser(
        description='Generate conversation labels'
    )
    parser.add_argument('--conv-list', required=True)
    parser.add_argument('--conv-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--duration', type=float, default=32.0)
    parser.add_argument('--frame-rate', type=int, default=100,
                       help='Label frame rate (fps)')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 读取对话列表
    with open(args.conv_list, 'r') as f:
        conv_ids = [line.strip() for line in f]
    
    print(f"Generating labels for {len(conv_ids)} conversations...")
    
    for i, conv_id in enumerate(conv_ids):
        # 加载元数据
        meta_file = os.path.join(args.conv_dir, f'{conv_id}.json')
        
        if not os.path.exists(meta_file):
            print(f"Warning: metadata not found for {conv_id}")
            continue
        
        with open(meta_file, 'r') as f:
            metadata = json.load(f)
        
        # 生成标签
        labels = generate_speaker_labels(
            metadata,
            args.duration,
            args.frame_rate
        )
        
        # 保存标签
        output_file = os.path.join(args.output_dir, f'{conv_id}.json')
        with open(output_file, 'w') as f:
            json.dump(labels, f, indent = 2,separators=(', ', ': '), cls=NumpyEncoder)
        
        if (i + 1) % 1000 == 0:
            print(f"  Progress: {i + 1}/{len(conv_ids)}")
    
    print("Done!")


if __name__ == '__main__':
    main()
