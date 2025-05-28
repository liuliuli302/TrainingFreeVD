import torch
import copy
import sys
import os
from contextlib import contextmanager
from pathlib import Path
from PIL import Image
from typing import List, Union, Optional, TYPE_CHECKING

from src.config.config import LLMHandlerConfig
import warnings
warnings.filterwarnings("ignore")
# Suppress specific transformers warnings
warnings.filterwarnings("ignore", message=".*attention_mask.*")
warnings.filterwarnings("ignore", message=".*pad token.*")

# For local model
try:
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import tokenizer_image_token
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    from llava.conversation import conv_templates
    LLAVA_AVAILABLE = True
except ImportError:
    LLAVA_AVAILABLE = False

# For API calls
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class LLMHandler:
    """Handler for LLM interactions, supporting both local models and API calls"""
    
    def __init__(self, config: LLMHandlerConfig):
        """
        Initialize LLM handler
        
        Args:
            config: LLMHandlerConfig配置对象
        """
        self.config = config
        self.model_type = config.model_type
        self.model = None
        self.tokenizer = None
        self.image_processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if self.model_type == "local":
            self._init_local_model(config.pretrained, config.model_name, config.device_map)
        elif self.model_type == "api":
            self._init_api_client(config.api_key, config.base_url)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def _init_local_model(self, pretrained: str = "lmms-lab/LLaVA-Video-7B-Qwen2", 
                         model_name: str = "llava_qwen", device_map: str = "auto"):
        """Initialize local LLaVA model"""
        if not LLAVA_AVAILABLE:
            raise ImportError("LLaVA package not available. Please install it first.")
        
        print(f"Loading local model: {pretrained}")
        self.tokenizer, self.model, self.image_processor, self.max_length = load_pretrained_model(
            pretrained, None, model_name, torch_dtype="bfloat16", device_map=device_map
        )
        self.model = self.model.eval()
        self.conv_template = "qwen_1_5"
        print("Local model loaded successfully")
    
    def _init_api_client(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """Initialize API client"""
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package not available. Please install it first.")
        
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        print("API client initialized")
    
    def preprocess_images(self, image_paths: List[str]) -> Union[torch.Tensor, List[Image.Image]]:
        """Preprocess images for model input"""
        images = []
        for path in image_paths:
            images.append(Image.open(path).convert('RGB'))
        
        if self.model_type == "local":
            processed_images = self.image_processor.preprocess(
                images, return_tensors="pt"
            )["pixel_values"].to(self.device).bfloat16()
            return [processed_images]
        else:
            return images
    
    def generate_response(self, images: Union[torch.Tensor, List[Image.Image]], 
                         prompt: str, video_time: float, frame_time: str,
                         max_new_tokens: int = 500, temperature: float = 0) -> str:
        """Generate response from LLM"""
        if self.model_type == "local":
            return self._generate_local_response(
                images, prompt, video_time, frame_time, max_new_tokens, temperature
            )
        else:
            return self._generate_api_response(
                images, prompt, video_time, frame_time, max_new_tokens, temperature
            )
    
    def _generate_local_response(self, images: torch.Tensor, prompt: str, 
                               video_time: float, frame_time: str,
                               max_new_tokens: int, temperature: float) -> str:
        """Generate response using local model"""
        time_instruction = (
            f"The video lasts for {video_time:.2f} seconds, and {len(images[0])} frames "
            f"are uniformly sampled from it. These frames are located at {frame_time}. "
            f"Please answer the following questions related to this video."
        )
        
        question = DEFAULT_IMAGE_TOKEN + f"\n{time_instruction}\n{prompt}"
        conv = copy.deepcopy(conv_templates[self.conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()
        
        input_ids = tokenizer_image_token(
            prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            cont = self.model.generate(
                input_ids,
                images=images,
                modalities=["video"],
                do_sample=temperature > 0,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
            )
        
        text_outputs = self.tokenizer.batch_decode(
            cont, skip_special_tokens=True
        )[0].strip()
        
        return text_outputs
    
    def _generate_api_response(self, images: List[Image.Image], prompt: str,
                             video_time: float, frame_time: str,
                             max_new_tokens: int, temperature: float) -> str:
        """Generate response using API"""
        # This is a placeholder for API implementation
        # You would need to implement the specific API call based on your provider
        time_instruction = (
            f"The video lasts for {video_time:.2f} seconds, and {len(images)} frames "
            f"are uniformly sampled from it. These frames are located at {frame_time}. "
            f"Please answer the following questions related to this video."
        )
        
        full_prompt = f"{time_instruction}\n{prompt}"
        
        # Example API call (adjust based on your API provider)
        response = self.client.chat.completions.create(
            model="gpt-4-vision-preview",  # Adjust model name
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": full_prompt},
                        # Add image content here based on API requirements
                    ]
                }
            ],
            max_tokens=max_new_tokens,
            temperature=temperature,
        )
        
        return response.choices[0].message.content
    
    def multi_turn_conversation(self, images: Union[torch.Tensor, List[Image.Image]],
                              prompts: List[str], video_time: float, frame_time: str,
                              max_new_tokens: int = 1000) -> List[str]:
        """Handle multi-turn conversation"""
        if self.model_type == "local":
            return self._multi_turn_local(images, prompts, video_time, frame_time, max_new_tokens)
        else:
            return self._multi_turn_api(images, prompts, video_time, frame_time, max_new_tokens)
    
    def _multi_turn_local(self, images: torch.Tensor, prompts: List[str],
                         video_time: float, frame_time: str, max_new_tokens: int) -> List[str]:
        """Multi-turn conversation for local model"""
        time_instruction = (
            f"The video lasts for {video_time:.2f} seconds, and {len(images[0])} frames "
            f"are uniformly sampled from it. These frames are located at {frame_time}. "
            f"Please answer the following questions related to this video."
        )
        
        conv = copy.deepcopy(conv_templates[self.conv_template])
        responses = []
        
        for i, prompt in enumerate(prompts):
            if i == 0:
                question = DEFAULT_IMAGE_TOKEN + f"\n{time_instruction}\n{prompt}"
            else:
                question = prompt
            
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            prompt_question = conv.get_prompt()
            
            input_ids = tokenizer_image_token(
                prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            ).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                cont = self.model.generate(
                    input_ids,
                    images=images,
                    modalities=["video"],
                    do_sample=False,
                    temperature=0,
                    max_new_tokens=max_new_tokens,
                )
            
            text_outputs = self.tokenizer.batch_decode(
                cont, skip_special_tokens=True
            )[0].strip()
            
            # Update conversation with response
            conv.messages[-1][1] = text_outputs
            responses.append(text_outputs)
        
        return responses
    
    def _multi_turn_api(self, images: List[Image.Image], prompts: List[str],
                       video_time: float, frame_time: str, max_new_tokens: int) -> List[str]:
        """Multi-turn conversation for API"""
        # Placeholder for API multi-turn implementation
        responses = []
        for prompt in prompts:
            response = self._generate_api_response(
                images, prompt, video_time, frame_time, max_new_tokens, 0
            )
            responses.append(response)
        return responses