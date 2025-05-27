"""
视频摘要Pipeline各组件配置类 
"""
import numpy as np


class ExtractorConfig:
    def __init__(
            self,
            model_name: str = "Salesforce/blip2-opt-2.7b",
            device: str = None,
            videos_dir: str = None,
            frames_dir: str = None,
            annotations_file: str = None,
            stride: int = 15,
            batch_size: int = 128
    ):
        self.model_name = model_name
        self.device = device
        self.videos_dir = videos_dir
        self.frames_dir = frames_dir
        self.annotations_file = annotations_file
        self.stride = stride
        self.batch_size = batch_size
        # 常量配置
        self.video_extensions = (
            '.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv')
        self.image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')


class DatasetBuilderConfig:
    def __init__(
        self,
        data_dir: str = None,
        save_dir: str = None,
        clip_length: int = 5
    ):
        self.data_dir = data_dir
        self.save_dir = save_dir
        self.clip_length = clip_length
        # 常量配置
        self.clip_types = {"turn": "00", "jump": "01"}
        self.dataset_prompt = (
            "You are a professional short film editor and director. "
            "Please score the frames divided based on theirs representativeness, "
            "diversity, and interest on a scale from 0 to 1. You may need to refer "
            "to the context for rating. And give the final score list like `[scores]`.\n "
            "without any extra text. You must output score."
        )


class CaptionerConfig:
    def __init__(
        self,
        model_name: str = "lmms-lab/LLaVA-Video-7B-Qwen2",
        max_frames: int = 64,
        conv_template: str = "qwen_1_5",
        video_folder: str = None,
        output_folder: str = None,
        prompt_templates: list = None
    ):
        self.model_name = model_name
        self.max_frames = max_frames
        self.conv_template = conv_template
        self.video_folder = video_folder
        self.output_folder = output_folder
        self.prompt_templates = prompt_templates or [
            "Summarize the main content and main events of the video in a concise and clear manner according to the order of events."
        ]


class LLMHandlerConfig:
    def __init__(
        self,
        model_type: str = "local",
        pretrained: str = "lmms-lab/LLaVA-Video-7B-Qwen2",
        model_name: str = "llava_qwen",
        device_map: str = "auto",
        api_key: str = None,
        base_url: str = None
    ):
        self.model_type = model_type
        self.pretrained = pretrained
        self.model_name = model_name
        self.device_map = device_map
        self.api_key = api_key
        self.base_url = base_url


class LLMQueryConfig:
    def __init__(
        self,
        dataset_dir: str = None,
        result_dir: str = None,
        scores_dir: str = None,
        frame_interval: int = 15,
        first_prompt: str = None,
        second_prompt: str = None
    ):
        self.dataset_dir = dataset_dir
        self.result_dir = result_dir
        self.scores_dir = scores_dir
        self.frame_interval = frame_interval
        self.first_prompt = first_prompt or (
            "You are a professional short film editor and director. "
            "Please score the frames divided based on theirs representativeness, "
            "diversity, and interest on a scale from 0 to 1. You may need to refer "
            "to the context for rating. And give the final score list like `[scores]`.\n "
            "without any extra text. You must output score."
        )
        self.second_prompt = second_prompt or (
            "Please provide a detailed analysis of each frame and give final scores."
        )
        # 常量配置
        self.score_pattern = r"Score:\s*(\d+\.\d+)"
        self.dataset_file_names = [
            "summe_dataset_jump.json",
            "summe_dataset_turn.json",
            "tvsum_dataset_jump.json",
            "tvsum_dataset_turn.json"
        ]
        self.hdf5_file_names = {
            "summe": "summe.h5",
            "tvsum": "tvsum.h5"
        }


class EvaluatorConfig:
    def __init__(
        self,
        dataset_dir: str = None,
        llm_score_file: str = None,
        similarity_scores_dir: str = None,
        output_dir: str = None,
        eval_method: str = "avg"
    ):
        self.dataset_dir = dataset_dir
        self.llm_score_file = llm_score_file
        self.similarity_scores_dir = similarity_scores_dir
        self.output_dir = output_dir
        self.eval_method = eval_method
        # 常量配置
        self.eval_methods = {
            "summe": "max",
            "tvsum": "avg"
        }
        self.alpha_range = [round(x, 1) for x in np.arange(0.1, 1.1, 0.1)]
        self.summary_ratio = 0.15


class UtilConfig:
    def __init__(
        self,
        visual_features_dir: str = None,
        text_features_dir: str = None,
        similarity_scores_dir: str = None,
        segment_num: int = 5,
        epsilon: float = 1e-8
    ):
        self.visual_features_dir = visual_features_dir
        self.text_features_dir = text_features_dir
        self.similarity_scores_dir = similarity_scores_dir
        self.segment_num = segment_num
        self.epsilon = epsilon


class VideoSummarizationPipelineConfig:
    def __init__(
        self,
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
        util_config: UtilConfig = None
    ):

        self.base_data_dir = base_data_dir
        self.base_output_dir = base_output_dir
        self.dataset_name = dataset_name
        self.exam_name = exam_name  # 保存实验名称

        # 全局常量配置 - 减少重复定义
        self.constants = {
            "dataset_file_names": [
                "summe_dataset_jump.json",
                "summe_dataset_turn.json",
                "tvsum_dataset_jump.json",
                "tvsum_dataset_turn.json"
            ],
            "hdf5_file_names": {
                "summe": "summe.h5",
                "tvsum": "tvsum.h5"
            },
            "similarity_score_types": [
                "max_p_max_m",
                "max_p_mean_m",
                "mean_p_max_m",
                "mean_p_mean_m"
            ],
            "llm_types": ["jump", "turn"],
            "dataset_mapping": {
                "summe": "SumMe",
                "tvsum": "TVSum"
            },
            "video_extensions": ('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv'),
            "image_extensions": ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        }

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
