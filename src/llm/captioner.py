import os
import torch
import warnings
import numpy as np
from typing import List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from config.config import CaptionerConfig
from decord import VideoReader, cpu
from tqdm import tqdm
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
import copy

warnings.filterwarnings("ignore")


class VideoCaptioner:
    """Video captioning class using LLaVA-Video model"""
    
    def __init__(self, config: 'CaptionerConfig'):
        """
        Initialize VideoCaptioner
        
        Args:
            config: CaptionerConfig配置对象
        """
        self.config = config
        self.model_name = config.model_name
        self.max_frames = config.max_frames
        self.conv_template = config.conv_template
        self.model = None
        self.tokenizer = None
        self.image_processor = None
        self.max_length = None
        
        # 使用配置中的默认提示词
        self.default_prompts = self.config.prompt_templates
    
    def load_model(self):
        """Load the pretrained model and processors"""
        if self.model is None:
            print(f"Loading model: {self.model_name}")
            self.tokenizer, self.model, self.image_processor, self.max_length = load_pretrained_model(
                self.model_name, 
                None, 
                "llava_qwen",
                torch_dtype="bfloat16", 
                device_map="auto"
            )
            self.model.eval()
            print("Model loaded successfully!")
    
    def load_video(self, video_path: str, fps: int = 1, force_sample: bool = True) -> Tuple[np.ndarray, str, float]:
        """
        Load and sample frames from video
        
        Args:
            video_path: Path to video file
            fps: Frames per second for sampling
            force_sample: Whether to force uniform sampling
            
        Returns:
            Tuple of (frames, frame_time_str, video_duration)
        """
        if self.max_frames == 0:
            return np.zeros((1, 336, 336, 3)), "0.00s", 0.0
            
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        total_frame_num = len(vr)
        video_time = total_frame_num / vr.get_avg_fps()
        fps = round(vr.get_avg_fps() / fps)
        frame_idx = [i for i in range(0, len(vr), fps)]
        frame_time = [i / fps for i in frame_idx]
        
        if len(frame_idx) > self.max_frames or force_sample:
            uniform_sampled_frames = np.linspace(
                0, total_frame_num - 1, self.max_frames, dtype=int)
            frame_idx = uniform_sampled_frames.tolist()
            frame_time = [i / vr.get_avg_fps() for i in frame_idx]
        
        frame_time_str = ",".join([f"{i:.2f}s" for i in frame_time])
        spare_frames = vr.get_batch(frame_idx).asnumpy()
        
        return spare_frames, frame_time_str, video_time
    
    @torch.no_grad()
    def caption_video(self, video_path: str, prompt_templates: Optional[List[str]] = None) -> List[str]:
        """
        Generate captions for a video
        
        Args:
            video_path: Path to video file
            prompt_templates: List of prompt templates to use
            
        Returns:
            List of generated captions
        """
        if self.model is None:
            self.load_model()
            
        if prompt_templates is None:
            prompt_templates = self.default_prompts
            
        # Load and process video
        video, frame_time, video_time = self.load_video(video_path)
        video = self.image_processor.preprocess(video, return_tensors="pt")["pixel_values"].cuda().bfloat16()
        video = [video]
        
        captions = []
        for template in prompt_templates:
            time_instruction = (
                f"The video lasts for {video_time:.2f} seconds, and {len(video[0])} frames are uniformly sampled from it. "
                f"These frames are located at {frame_time}. Please answer the following questions related to this video."
            )
            question = DEFAULT_IMAGE_TOKEN + f"\n{time_instruction}\n{template}"
            
            # Prepare conversation
            conv = copy.deepcopy(conv_templates[self.conv_template])
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            prompt_question = conv.get_prompt()
            
            input_ids = tokenizer_image_token(
                prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            ).unsqueeze(0).cuda()
            
            attention_mask = torch.ones_like(input_ids).cuda()
            
            # Generate caption
            cont = self.model.generate(
                input_ids,
                images=video,
                modalities=["video"],
                do_sample=False,
                temperature=0,
                max_new_tokens=4096,
                attention_mask=attention_mask
            )
            
            text_output = self.tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()
            captions.append(text_output)
        
        return captions
    
    def caption_single_video(self, video_path: str, output_path: Optional[str] = None, 
                           prompt_templates: Optional[List[str]] = None) -> List[str]:
        """
        Caption a single video and optionally save to file
        
        Args:
            video_path: Path to video file
            output_path: Optional path to save caption text file
            prompt_templates: List of prompt templates to use
            
        Returns:
            List of generated captions
        """
        captions = self.caption_video(video_path, prompt_templates)
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                for caption in captions:
                    f.write(caption + '\n')
        
        return captions
    
    def caption_video_folder(self, video_folder=None, output_folder=None, prompt_templates=None):
        """
        Caption all videos in a folder
        
        Args:
            video_folder: Path to folder containing videos（可选，默认使用config中的参数）
            output_folder: Path to folder for saving caption files（可选，默认使用config中的参数）
            prompt_templates: List of prompt templates to use（可选，默认使用config中的参数）
        """
        video_folder = video_folder or self.config.video_folder
        output_folder = output_folder or self.config.output_folder
        prompt_templates = prompt_templates or self.config.prompt_templates
        
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        # 使用配置中的视频扩展名
        video_files = [f for f in os.listdir(video_folder) if f.lower().endswith(self.config.video_extensions)]
        
        if not video_files:
            print(f"No video files found in {video_folder}")
            return
        
        print(f"Found {len(video_files)} video files to process")
        
        for video_name in tqdm(video_files, desc="Processing videos"):
            video_path = os.path.join(video_folder, video_name)
            txt_name = os.path.splitext(video_name)[0] + '.txt'
            txt_path = os.path.join(output_folder, txt_name)
            
            try:
                captions = self.caption_video(video_path, prompt_templates)
                with open(txt_path, 'w', encoding='utf-8') as f:
                    for caption in captions:
                        f.write(caption + '\n')
                print(f"✓ Processed: {video_name}")
            except Exception as e:
                print(f"✗ Error processing {video_name}: {str(e)}")
    
    def set_prompts(self, prompts: List[str]):
        """Set custom prompt templates"""
        self.default_prompts = prompts
    
    def add_prompt(self, prompt: str):
        """Add a new prompt template"""
        self.default_prompts.append(prompt)
    
    def clear_prompts(self):
        """Clear all prompt templates"""
        self.default_prompts = []


# Example usage
if __name__ == "__main__":
    # Initialize captioner
    captioner = VideoCaptioner()
    
    # Example 1: Caption single video
    video_path = "path/to/your/video.mp4"
    captions = captioner.caption_single_video(video_path, "output.txt")
    print("Generated captions:", captions)
    
    # Example 2: Caption all videos in folder
    video_folder = "path/to/video/folder"
    output_folder = "path/to/output/folder"
    captioner.caption_video_folder(video_folder, output_folder)
    
    # Example 3: Use custom prompts
    custom_prompts = [
        "Describe the main activities happening in this video.",
        "What objects and people can you see in this video?",
        "Explain the sequence of events in this video."
    ]
    captioner.set_prompts(custom_prompts)
    captions = captioner.caption_single_video(video_path)