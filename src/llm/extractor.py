import argparse
import os
from pathlib import Path
import cv2
import torch
import numpy as np
import math
from PIL import Image
from transformers import Blip2Processor, Blip2Model
from decord import VideoReader, cpu
from tqdm import tqdm
from typing import TYPE_CHECKING

from src.config.config import ExtractorConfig


class Extractor:
    def __init__(self, config: ExtractorConfig):
        """
        初始化提取器
        
        Args:
            config: ExtractorConfig配置对象
        """
        self.config = config
        self.device = config.device if config.device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # 加载BLIP-2模型和处理器
        self.processor = Blip2Processor.from_pretrained(config.model_name)
        self.model = Blip2Model.from_pretrained(
            config.model_name, torch_dtype=torch.float32)
        self.model.to(self.device)
        
    def extract_frames(self, video_path=None, frames_dir=None):
        """
        从视频中提取帧
        
        Args:
            video_path: 视频文件路径（可选，默认使用config中的参数）
            frames_dir: 帧保存目录（可选，默认使用config中的参数）
            
        Returns:
            tuple: (视频名称, 帧数量)
        """
        video_path = video_path or self.config.videos_dir
        frames_dir = frames_dir or self.config.frames_dir
        
        video_name = Path(video_path).stem
        video_frames_dir = os.path.join(frames_dir, video_name)
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

    def extract_frames_from_directory(self, videos_dir=None, frames_dir=None, annotations_file=None):
        """
        批量从视频目录提取帧
        
        Args:
            videos_dir: 视频目录（可选，默认使用config中的参数）
            frames_dir: 帧保存目录（可选，默认使用config中的参数）
            annotations_file: 标注文件路径（可选，默认使用config中的参数）
        """
        videos_dir = videos_dir or self.config.videos_dir
        frames_dir = frames_dir or self.config.frames_dir
        annotations_file = annotations_file or self.config.annotations_file
        
        os.makedirs(frames_dir, exist_ok=True)
        os.makedirs(os.path.dirname(annotations_file), exist_ok=True)
        
        with open(annotations_file, "w") as f:
            for video_file in os.listdir(videos_dir):
                if video_file.endswith((".avi", ".mp4")):
                    video_path = os.path.join(videos_dir, video_file)
                    video_name, num_frames = self.extract_frames(video_path, frames_dir)
                    f.write(f"{video_name} 0 {num_frames - 1} 0\n")

    @torch.no_grad()
    def extract_visual_features(self, video_folder=None, output_folder=None, stride=None, batch_size=None):
        """
        提取视频的视觉特征
        
        Args:
            video_folder: 视频文件夹路径（可选，默认使用config中的参数）
            output_folder: 特征保存文件夹（可选，默认使用config中的参数）
            stride: 帧采样步长（可选，默认使用config中的参数）
            batch_size: 批处理大小（可选，默认使用config中的参数）
        """
        video_folder = video_folder or self.config.video_folder or self.config.videos_dir
        output_folder = output_folder or self.config.visual_features_dir or self.config.output_folder
        stride = stride or self.config.stride
        batch_size = batch_size or self.config.batch_size
        
        video_folder = Path(video_folder)
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)

        # 获取视频文件列表
        video_files = [f for f in video_folder.iterdir() 
                      if f.suffix in ['.mp4', '.avi', '.mov']]

        for video_file in tqdm(video_files, desc="Processing videos", unit="video"):
            vr = VideoReader(str(video_file), ctx=cpu(0))
            frame_count = len(vr)

            # 生成按照stride采样的帧索引
            indices = list(range(0, frame_count, stride))
            frames = vr.get_batch(indices).asnumpy()

            all_features = []
            num_batches = math.ceil(len(frames) / batch_size)

            for batch_idx in tqdm(range(num_batches), 
                                desc=f"Processing batches in {video_file.name}", 
                                unit="batch", leave=False):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(frames))
                batch_frames = frames[start_idx:end_idx]

                batch_images = [Image.fromarray(frame) for frame in batch_frames]
                inputs = self.processor(images=batch_images, return_tensors="pt").to(
                    self.device, torch.float32)

                # 手动执行 vision -> qformer -> language projection
                vision_outputs = self.model.vision_model(pixel_values=inputs.pixel_values)
                image_embeds = vision_outputs[0]

                image_attention_mask = torch.ones(
                    image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)
                query_tokens = self.model.query_tokens.expand(image_embeds.shape[0], -1, -1)

                qformer_outputs = self.model.qformer(
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_attention_mask,
                    return_dict=True
                )
                query_output = qformer_outputs.last_hidden_state
                projected_features = self.model.language_projection(query_output)

                if projected_features.dtype != image_embeds.dtype:
                    projected_features = projected_features.to(image_embeds.dtype)

                batch_features = projected_features.cpu().numpy()
                all_features.append(batch_features)

            final_features = np.concatenate(all_features, axis=0)
            feature_file = output_folder / f'{video_file.stem}.npy'
            np.save(feature_file, final_features)

    @torch.no_grad()
    def extract_text_features(self, text_folder=None, output_folder=None):
        """
        提取文本特征
        
        Args:
            text_folder: 文本文件夹路径（可选，默认使用config中的参数）
            output_folder: 特征保存文件夹（可选，默认使用config中的参数）
        """
        text_folder = text_folder or self.config.text_folder
        output_folder = output_folder or self.config.text_features_dir or self.config.output_folder
        
        text_folder = Path(text_folder)
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)

        text_files = [f for f in text_folder.iterdir() if f.suffix == ".txt"]

        for text_file in tqdm(text_files, desc="Processing text files", unit="file"):
            with open(text_file, 'r', encoding='utf-8') as f:
                text = f.read().strip()

            # 处理文本为 input_ids
            inputs = self.processor(text=[text], return_tensors="pt").to(self.device)
            input_ids = inputs["input_ids"]
            
            # 提取词嵌入
            inputs_embeds = self.model.language_model.get_input_embeddings()(input_ids)
            text_features = inputs_embeds.squeeze(0).cpu().numpy()

            # 保存
            feature_file = output_folder / f"{text_file.stem}_text.npy"
            np.save(feature_file, text_features)

    def extract_single_video_features(self, video_path, stride=1, batch_size=16):
        """
        提取单个视频的特征（返回numpy数组，不保存文件）
        
        Args:
            video_path: 视频文件路径
            stride: 帧采样步长
            batch_size: 批处理大小
            
        Returns:
            numpy.ndarray: 视频特征数组
        """
        vr = VideoReader(str(video_path), ctx=cpu(0))
        frame_count = len(vr)
        
        indices = list(range(0, frame_count, stride))
        frames = vr.get_batch(indices).asnumpy()
        
        all_features = []
        num_batches = math.ceil(len(frames) / batch_size)
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(frames))
            batch_frames = frames[start_idx:end_idx]
            
            batch_images = [Image.fromarray(frame) for frame in batch_frames]
            inputs = self.processor(images=batch_images, return_tensors="pt").to(
                self.device, torch.float32)
            
            vision_outputs = self.model.vision_model(pixel_values=inputs.pixel_values)
            image_embeds = vision_outputs[0]
            
            image_attention_mask = torch.ones(
                image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)
            query_tokens = self.model.query_tokens.expand(image_embeds.shape[0], -1, -1)
            
            qformer_outputs = self.model.qformer(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_attention_mask,
                return_dict=True
            )
            query_output = qformer_outputs.last_hidden_state
            projected_features = self.model.language_projection(query_output)
            
            if projected_features.dtype != image_embeds.dtype:
                projected_features = projected_features.to(image_embeds.dtype)
            
            batch_features = projected_features.cpu().numpy()
            all_features.append(batch_features)
        
        return np.concatenate(all_features, axis=0)

    def extract_single_text_features(self, text):
        """
        提取单个文本的特征（返回numpy数组，不保存文件）
        
        Args:
            text: 文本内容
            
        Returns:
            numpy.ndarray: 文本特征数组
        """
        inputs = self.processor(text=[text], return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        
        inputs_embeds = self.model.language_model.get_input_embeddings()(input_ids)
        text_features = inputs_embeds.squeeze(0).cpu().numpy()
        
        return text_features


def parse_args():
    parser = argparse.ArgumentParser(description="Feature extraction utility")
    parser.add_argument('--mode', type=str, required=True, 
                       choices=['frames', 'visual', 'text'],
                       help="Extraction mode: frames, visual, or text")
    
    # 通用参数
    parser.add_argument('--input_folder', type=str, required=True,
                       help="Input folder path")
    parser.add_argument('--output_folder', type=str, required=True,
                       help="Output folder path")
    
    # 帧提取参数
    parser.add_argument('--annotations_file', type=str,
                       help="Annotations file path (for frames mode)")
    
    # 视觉特征提取参数
    parser.add_argument('--stride', type=int, default=1,
                       help="Frame sampling stride (for visual mode)")
    parser.add_argument('--batch_size', type=int, default=16,
                       help="Batch size (for visual mode)")
    
    return parser.parse_args()


def main():
    args = parse_args()
    extractor = Extractor()
    
    if args.mode == 'frames':
        if not args.annotations_file:
            raise ValueError("annotations_file is required for frames mode")
        extractor.extract_frames_from_directory(
            args.input_folder, args.output_folder, args.annotations_file)
    
    elif args.mode == 'visual':
        extractor.extract_visual_features(
            args.input_folder, args.output_folder, args.stride, args.batch_size)
    
    elif args.mode == 'text':
        extractor.extract_text_features(args.input_folder, args.output_folder)


if __name__ == "__main__":
    main()