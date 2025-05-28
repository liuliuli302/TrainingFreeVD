import os
import cv2
from pathlib import Path


class FrameExtractor:
    """独立的帧提取器类"""
    
    def __init__(self):
        """初始化帧提取器"""
        pass
    
    def _check_frames_exist(self, video_frames_dir):
        """
        检查指定目录中是否已经存在帧文件
        
        Args:
            video_frames_dir: 视频帧目录
            
        Returns:
            tuple: (是否存在帧, 现有帧数量)
        """
        if not os.path.exists(video_frames_dir):
            return False, 0
        
        # 检查目录中的jpg文件数量
        frame_files = [f for f in os.listdir(video_frames_dir) 
                      if f.endswith('.jpg') and f[:-4].isdigit()]
        
        if len(frame_files) > 0:
            return True, len(frame_files)
        
        return False, 0
    
    def extract_frames(self, video_path, frames_dir, skip_if_exists=True):
        """
        从视频中提取帧
        
        Args:
            video_path: 视频文件路径
            frames_dir: 帧保存目录
            skip_if_exists: 如果帧已存在是否跳过提取，默认为True
            
        Returns:
            tuple: (视频名称, 帧数量)
        """
        video_name = Path(video_path).stem
        video_frames_dir = os.path.join(frames_dir, video_name)
        
        # 检查是否已经存在帧
        frames_exist, existing_frame_count = self._check_frames_exist(video_frames_dir)
        
        if skip_if_exists and frames_exist:
            print(f"帧已存在，跳过提取: {video_frames_dir} (共 {existing_frame_count} 帧)")
            return video_name, existing_frame_count
        
        os.makedirs(video_frames_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_path = os.path.join(video_frames_dir, f"{frame_count:06d}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_count += 1
            
        cap.release()
        print(f"Extracted {frame_count} frames from {video_path} to {video_frames_dir}")
        return video_name, frame_count

    def extract_frames_from_directory(self, videos_dir, frames_dir, annotations_file, skip_if_exists=True):
        """
        批量从视频目录提取帧
        
        Args:
            videos_dir: 视频目录
            frames_dir: 帧保存目录
            annotations_file: 标注文件路径
            skip_if_exists: 如果帧已存在是否跳过提取，默认为True
        """
        os.makedirs(frames_dir, exist_ok=True)
        os.makedirs(os.path.dirname(annotations_file), exist_ok=True)
        
        # 读取已有的注释信息（如果存在）
        existing_annotations = {}
        if os.path.exists(annotations_file):
            with open(annotations_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        existing_annotations[parts[0]] = line.strip()
        
        updated_annotations = {}
        
        for video_file in os.listdir(videos_dir):
            if video_file.endswith((".avi", ".mp4")):
                video_path = os.path.join(videos_dir, video_file)
                video_name = Path(video_path).stem
                
                # 如果跳过已存在的帧，且该视频已在注释中，检查帧是否确实存在
                if skip_if_exists and video_name in existing_annotations:
                    video_frames_dir = os.path.join(frames_dir, video_name)
                    frames_exist, frame_count = self._check_frames_exist(video_frames_dir)
                    if frames_exist:
                        print(f"视频 {video_name} 的帧已存在，跳过提取")
                        updated_annotations[video_name] = existing_annotations[video_name]
                        continue
                
                # 提取帧
                video_name, num_frames = self.extract_frames(video_path, frames_dir, skip_if_exists)
                updated_annotations[video_name] = f"{video_name} 0 {num_frames - 1} 0"
        
        # 写入更新后的注释文件
        with open(annotations_file, "w") as f:
            for video_name in sorted(updated_annotations.keys()):
                f.write(f"{updated_annotations[video_name]}\n")