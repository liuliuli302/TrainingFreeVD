import argparse
import os
from pathlib import Path
import torch
import numpy as np
import math
from PIL import Image
from transformers import Blip2Processor, Blip2Model
from decord import VideoReader, cpu
from tqdm import tqdm
from typing import TYPE_CHECKING

from src.config.config import ExtractorConfig
from src.llm.frame_extractor import FrameExtractor


class Extractor:
    def __init__(self, config: ExtractorConfig):
        """
        初始化提取器
        
        Args:
            config: ExtractorConfig配置对象
        """
        self.config = config
        self.device = config.device if config.device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # 延迟加载模型和处理器
        self.processor = None
        self.model = None
        self._model_loaded = False
        
        # 初始化独立的帧提取器
        self.frame_extractor = FrameExtractor()
    
    def _load_model(self):
        """延迟加载BLIP-2模型和处理器"""
        if not self._model_loaded:
            print(f"Loading BLIP-2 model: {self.config.model_name}")
            self.processor = Blip2Processor.from_pretrained(self.config.model_name)
            self.model = Blip2Model.from_pretrained(
                self.config.model_name, torch_dtype=torch.float16)
            self.model.to(self.device)
            self._model_loaded = True
            print("BLIP-2 model loaded successfully!")
        
    def extract_frames(self, video_path=None, frames_dir=None, skip_if_exists=True):
        """
        从视频中提取帧（使用独立的帧提取器）
        
        Args:
            video_path: 视频文件路径（可选，默认使用config中的参数）
            frames_dir: 帧保存目录（可选，默认使用config中的参数）
            skip_if_exists: 如果帧已存在是否跳过提取，默认为True
            
        Returns:
            tuple: (视频名称, 帧数量)
        """
        video_path = video_path or self.config.videos_dir
        frames_dir = frames_dir or self.config.frames_dir
        
        return self.frame_extractor.extract_frames(video_path, frames_dir, skip_if_exists)

    def extract_frames_from_directory(self, videos_dir=None, frames_dir=None, annotations_file=None, skip_if_exists=True):
        """
        批量从视频目录提取帧（使用独立的帧提取器）
        
        Args:
            videos_dir: 视频目录（可选，默认使用config中的参数）
            frames_dir: 帧保存目录（可选，默认使用config中的参数）
            annotations_file: 标注文件路径（可选，默认使用config中的参数）
            skip_if_exists: 如果帧已存在是否跳过提取，默认为True
        """
        videos_dir = videos_dir or self.config.videos_dir
        frames_dir = frames_dir or self.config.frames_dir
        annotations_file = annotations_file or self.config.annotations_file
        
        self.frame_extractor.extract_frames_from_directory(videos_dir, frames_dir, annotations_file, skip_if_exists)

    @torch.no_grad()
    def extract_visual_features(self, video_folder=None, frames_root_dir=None, output_folder=None, stride=None, batch_size=None):
        """
        提取视频的视觉特征
        
        Args:
            video_folder: 视频文件夹路径（可选，默认使用config中的参数, 用于确定要处理哪些视频）
            frames_root_dir: 提取的帧文件存放的根目录 (可选, 默认使用config.frames_dir)
            output_folder: 特征保存文件夹（可选，默认使用config中的参数）
            stride: 帧采样步长（可选，默认使用config中的参数）
            batch_size: 批处理大小（可选，默认使用config中的参数）
        """
        # 只有在需要时才加载模型
        self._load_model()
        
        video_folder = Path(video_folder or self.config.video_folder or self.config.videos_dir)
        frames_root_dir = Path(frames_root_dir or self.config.frames_dir)
        output_folder = Path(output_folder or self.config.visual_features_dir or self.config.output_folder)
        stride = stride or self.config.stride
        batch_size = batch_size or self.config.batch_size
        
        output_folder.mkdir(parents=True, exist_ok=True)

        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
        video_files = [f for f in video_folder.iterdir() if f.suffix.lower() in video_extensions]

        for video_file in tqdm(video_files, desc="Processing videos for visual features", unit="video"):
            video_stem = video_file.stem
            feature_file = output_folder / f'{video_stem}.npy'
            
            if feature_file.exists():
                tqdm.write(f"Visual features for {video_file.name} already exist. Skipping.")
                continue

            current_video_frames_dir = frames_root_dir / video_stem
            if not current_video_frames_dir.exists():
                tqdm.write(f"Frame directory {current_video_frames_dir} not found for {video_file.name}. Skipping.")
                continue

            frame_files = sorted([f for f in current_video_frames_dir.iterdir() if f.suffix.lower() == '.jpg'])
            
            if not frame_files:
                tqdm.write(f"No frames found in {current_video_frames_dir} for {video_file.name}. Skipping.")
                continue
            
            frames_to_process = frame_files[::stride]

            if not frames_to_process:
                tqdm.write(f"No frames to process for {video_file.name} after applying stride. Skipping.")
                continue

            all_features = []
            num_batches = math.ceil(len(frames_to_process) / batch_size)

            try:
                for batch_idx in tqdm(range(num_batches), 
                                    desc=f"Processing batches in {video_file.name}", 
                                    unit="batch", leave=False):
                    start_idx = batch_idx * batch_size
                    end_idx = min(start_idx + batch_size, len(frames_to_process))
                    batch_frame_files = frames_to_process[start_idx:end_idx]

                    if not batch_frame_files:
                        continue

                    batch_images = []
                    for frame_path in batch_frame_files:
                        try:
                            img = Image.open(frame_path).convert("RGB")
                            batch_images.append(img)
                        except Exception as e:
                            tqdm.write(f"Error loading frame {frame_path} for {video_file.name}: {e}. Skipping frame.")
                    
                    if not batch_images:
                        tqdm.write(f"No images loaded for batch {batch_idx} in {video_file.name}. Skipping batch.")
                        continue
                    
                    # Process images and move pixel_values to device
                    processed_inputs = self.processor(images=batch_images, return_tensors="pt")
                    pixel_values = processed_inputs.pixel_values.to(self.device, torch.float16)

                    # 手动执行 vision -> qformer -> language projection
                    vision_outputs = self.model.vision_model(pixel_values=pixel_values)
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
                np.save(feature_file, final_features)
            except Exception as e:
                tqdm.write(f"Error processing video {video_file.name} for visual features: {type(e).__name__} - {e}. Skipping.")
                if hasattr(e, 'input_meta'):
                    tqdm.write(f"Meta tensor info (if available): {e.input_meta}")
                continue

    @torch.no_grad()
    def extract_text_features(self, text_folder=None, output_folder=None):
        """
        提取文本特征
        
        Args:
            text_folder: 文本文件夹路径（可选，默认使用config中的参数）
            output_folder: 特征保存文件夹（可选，默认使用config中的参数）
        """
        # 只有在需要时才加载模型
        self._load_model()
        
        text_folder = text_folder or self.config.text_folder
        output_folder = output_folder or self.config.text_features_dir or self.config.output_folder
        
        text_folder = Path(text_folder)
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)

        text_files = [f for f in text_folder.iterdir() if f.suffix == ".txt"]

        for text_file in tqdm(text_files, desc="Processing text files", unit="file"):
            feature_file = output_folder / f"{text_file.stem}_text.npy"
            if feature_file.exists():
                tqdm.write(f"Text features for {text_file.name} already exist. Skipping.")
                continue

            with open(text_file, 'r', encoding='utf-8') as f:
                text = f.read().strip()

            # 处理文本为 input_ids
            inputs = self.processor(text=[text], return_tensors="pt").to(self.device)
            input_ids = inputs["input_ids"]
            
            # 提取词嵌入
            inputs_embeds = self.model.language_model.get_input_embeddings()(input_ids)
            text_features = inputs_embeds.squeeze(0).cpu().numpy()

            # 保存
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
        # 只有在需要时才加载模型
        self._load_model()
        
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
                self.device, torch.float16)
            
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
        # 只有在需要时才加载模型
        self._load_model()
        
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
    
    # 为命令行模式创建临时配置
    from src.config.config import ExtractorConfig
    config = ExtractorConfig()
    
    extractor = Extractor(config)
    
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