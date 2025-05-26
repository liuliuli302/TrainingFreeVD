"""
视频摘要Pipeline各组件配置类 
"""
from util.constant import (
    CLIP_TYPES, VIDEO_EXTENSIONS, SCORE_PATTERN, DEFAULT_PROMPT_TEMPLATES,
    DEFAULT_DATASET_PROMPT, DEFAULT_FIRST_PROMPT, DEFAULT_SECOND_PROMPT,
    DATASET_FILE_NAMES, HDF5_FILE_NAMES, EVAL_METHODS, SIMILARITY_SCORE_TYPES,
    LLM_TYPES, ALPHA_RANGE, DEFAULT_FRAME_INTERVAL, DEFAULT_MAX_FRAMES,
    DEFAULT_CONV_TEMPLATE, DEFAULT_STRIDE, DEFAULT_BATCH_SIZE, IMAGE_EXTENSIONS,
    DEFAULT_SUMMARY_RATIO, PATH_SUFFIXES, DATASET_MAPPING, EPSILON
)

class ExtractorConfig:
    def __init__(self, 
                 model_name: str = "Salesforce/blip2-opt-2.7b",
                 device: str = None,
                 videos_dir: str = None,
                 frames_dir: str = None,
                 annotations_file: str = None,
                 stride: int = DEFAULT_STRIDE,
                 batch_size: int = DEFAULT_BATCH_SIZE,
                 video_folder: str = None,
                 output_folder: str = None,
                 text_folder: str = None,
                 visual_features_dir: str = None,
                 text_features_dir: str = None):
        self.model_name = model_name
        self.device = device
        self.videos_dir = videos_dir
        self.frames_dir = frames_dir
        self.annotations_file = annotations_file
        self.stride = stride
        self.batch_size = batch_size
        self.video_folder = video_folder
        self.output_folder = output_folder
        self.text_folder = text_folder
        self.visual_features_dir = visual_features_dir
        self.text_features_dir = text_features_dir
        # 常量配置
        self.video_extensions = VIDEO_EXTENSIONS
        self.image_extensions = IMAGE_EXTENSIONS


class DatasetBuilderConfig:
    def __init__(self,
                 data_dir: str = None,
                 save_dir: str = None,
                 clip_length: int = 5):
        self.data_dir = data_dir
        self.save_dir = save_dir
        self.clip_length = clip_length
        # 常量配置
        self.clip_types = CLIP_TYPES
        self.dataset_prompt = DEFAULT_DATASET_PROMPT
        self.dataset_file_names = DATASET_FILE_NAMES
        self.hdf5_file_names = HDF5_FILE_NAMES
        self.path_suffixes = PATH_SUFFIXES


class CaptionerConfig:
    def __init__(self,
                 model_name: str = "lmms-lab/LLaVA-Video-7B-Qwen2",
                 max_frames: int = DEFAULT_MAX_FRAMES,
                 conv_template: str = DEFAULT_CONV_TEMPLATE,
                 video_folder: str = None,
                 output_folder: str = None,
                 prompt_templates: list = None):
        self.model_name = model_name
        self.max_frames = max_frames
        self.conv_template = conv_template
        self.video_folder = video_folder
        self.output_folder = output_folder
        self.prompt_templates = prompt_templates or DEFAULT_PROMPT_TEMPLATES.copy()
        # 常量配置
        self.video_extensions = VIDEO_EXTENSIONS


class LLMHandlerConfig:
    def __init__(self,
                 model_type: str = "local",
                 pretrained: str = "lmms-lab/LLaVA-Video-7B-Qwen2",
                 model_name: str = "llava_qwen",
                 device_map: str = "auto",
                 api_key: str = None,
                 base_url: str = None):
        self.model_type = model_type
        self.pretrained = pretrained
        self.model_name = model_name
        self.device_map = device_map
        self.api_key = api_key
        self.base_url = base_url


class LLMQueryConfig:
    def __init__(self,
                 dataset_dir: str = None,
                 result_dir: str = None,
                 scores_dir: str = None,
                 frame_interval: int = DEFAULT_FRAME_INTERVAL,
                 first_prompt: str = None,
                 second_prompt: str = None):
        self.dataset_dir = dataset_dir
        self.result_dir = result_dir
        self.scores_dir = scores_dir
        self.frame_interval = frame_interval
        self.first_prompt = first_prompt or DEFAULT_FIRST_PROMPT
        self.second_prompt = second_prompt or DEFAULT_SECOND_PROMPT
        # 常量配置
        self.score_pattern = SCORE_PATTERN
        self.dataset_file_names = DATASET_FILE_NAMES
        self.hdf5_file_names = HDF5_FILE_NAMES


class EvaluatorConfig:
    def __init__(self,
                 dataset_dir: str = None,
                 llm_score_file: str = None,
                 similarity_scores_dir: str = None,
                 output_dir: str = None,
                 eval_method: str = "avg"):
        self.dataset_dir = dataset_dir
        self.llm_score_file = llm_score_file
        self.similarity_scores_dir = similarity_scores_dir
        self.output_dir = output_dir
        self.eval_method = eval_method
        # 常量配置
        self.eval_methods = EVAL_METHODS
        self.similarity_score_types = SIMILARITY_SCORE_TYPES
        self.llm_types = LLM_TYPES
        self.alpha_range = ALPHA_RANGE
        self.summary_ratio = DEFAULT_SUMMARY_RATIO
        self.dataset_mapping = DATASET_MAPPING
        self.hdf5_file_names = HDF5_FILE_NAMES


class UtilConfig:
    def __init__(self,
                 visual_features_dir: str = None,
                 text_features_dir: str = None,
                 similarity_scores_dir: str = None,
                 segment_num: int = 5,
                 epsilon: float = EPSILON):
        self.visual_features_dir = visual_features_dir
        self.text_features_dir = text_features_dir
        self.similarity_scores_dir = similarity_scores_dir
        self.segment_num = segment_num
        self.epsilon = epsilon
        # Constants from constants.py
        self.similarity_score_types = SIMILARITY_SCORE_TYPES


class VideoSummarizationPipelineConfig:
    def __init__(self,
                 # 基础路径配置
                 base_data_dir: str = None,
                 base_output_dir: str = None,
                 dataset_name: str = "TVSum",
                 exam_name: str = None,  # 新增实验名称参数
                 
                 # 各组件特定配置
                 extractor_config: ExtractorConfig = None,
                 dataset_builder_config: DatasetBuilderConfig = None,
                 captioner_config: CaptionerConfig = None,
                 llm_handler_config: LLMHandlerConfig = None,
                 llm_query_config: LLMQueryConfig = None,
                 evaluator_config: EvaluatorConfig = None,
                 util_config: UtilConfig = None):
        
        self.base_data_dir = base_data_dir
        self.base_output_dir = base_output_dir
        self.dataset_name = dataset_name
        self.exam_name = exam_name  # 保存实验名称
        
        # 初始化各组件配置，如果没有提供则使用默认配置
        self.extractor_config = extractor_config or ExtractorConfig()
        self.dataset_builder_config = dataset_builder_config or DatasetBuilderConfig()
        self.captioner_config = captioner_config or CaptionerConfig()
        self.llm_handler_config = llm_handler_config or LLMHandlerConfig()
        self.llm_query_config = llm_query_config or LLMQueryConfig()
        self.evaluator_config = evaluator_config or EvaluatorConfig()
        self.util_config = util_config or UtilConfig()
        
        # 设置路径
        self._setup_paths()
    
    def _setup_paths(self):
        """根据基础配置自动设置各个组件的路径"""
        if self.base_data_dir and self.base_output_dir:
            # Extractor配置路径
            self.extractor_config.videos_dir = f"{self.base_data_dir}/{self.dataset_name}/videos"
            self.extractor_config.frames_dir = f"{self.base_data_dir}/{self.dataset_name}/frames"
            self.extractor_config.annotations_file = f"{self.base_data_dir}/{self.dataset_name}/annotations/test.txt"
            
            # DatasetBuilder配置路径
            self.dataset_builder_config.data_dir = self.base_data_dir
            self.dataset_builder_config.save_dir = f"{self.base_output_dir}/dataset"
            
            # Captioner配置路径
            self.captioner_config.video_folder = f"{self.base_data_dir}/{self.dataset_name}/videos"
            self.captioner_config.output_folder = f"{self.base_data_dir}/{self.dataset_name}/captions"
            
            # LLMQuery配置路径
            self.llm_query_config.dataset_dir = f"{self.base_output_dir}/dataset"
            self.llm_query_config.result_dir = f"{self.base_output_dir}/dataset/result/raw"
            self.llm_query_config.scores_dir = f"{self.base_output_dir}/dataset/result/scores"
            
            # Evaluator配置路径
            self.evaluator_config.dataset_dir = self.base_data_dir
            self.evaluator_config.llm_score_file = f"{self.base_output_dir}/dataset/result/scores/raw_llm_out_scores.json"
            self.evaluator_config.similarity_scores_dir = f"{self.base_output_dir}/dataset/result/similarity_scores"
            self.evaluator_config.output_dir = f"{self.base_output_dir}/dataset/result/f1score"
            
            # Util配置路径
            self.util_config.visual_features_dir = f"{self.base_data_dir}/{self.dataset_name}/features/visual"
            self.util_config.text_features_dir = f"{self.base_data_dir}/{self.dataset_name}/features/text"
            self.util_config.similarity_scores_dir = f"{self.base_output_dir}/dataset/result/similarity_scores"
