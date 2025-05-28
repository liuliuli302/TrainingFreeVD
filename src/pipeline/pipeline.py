"""
视频摘要生成Pipeline
"""
import gc
import logging
from pathlib import Path
from src.config.config import VideoSummarizationPipelineConfig
from src.llm.extractor import Extractor
from src.llm.frame_extractor import FrameExtractor
from src.dataset.dataset_builder import DatasetBuilder
from src.llm.captioner import VideoCaptioner
from src.llm.llm_handler import LLMHandler
from src.llm.llm_query import LLMQuery
from src.eval.evaluator import VideoSummaryEvaluator
from src.util.util import process_all


class VideoSummarizationPipeline:
    def __init__(self, config: VideoSummarizationPipelineConfig):
        self.config = config
        self.logger = self._setup_logger()
        
        # 记录实验信息
        if self.config.exam_name:
            self.logger.info(f"Initializing experiment: {self.config.exam_name}")
            self.logger.info(f"Experiment directory: {self.config.base_output_dir}")
        
        # 创建实验信息文件
        self._create_experiment_info_file()
    
    def _setup_logger(self):
        """设置日志记录器，将日志同时输出到控制台和实验目录的日志文件"""
        # 创建日志格式
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # 创建根logger
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        # 清除可能存在的处理器
        logger.handlers.clear()
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # 如果有实验目录，添加文件处理器
        if self.config.base_output_dir:
            log_file = Path(self.config.base_output_dir) / f"{self.config.exam_name}_pipeline.log"
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
        return logger
    
    def _create_experiment_info_file(self):
        """在实验目录中创建实验信息文件"""
        if not self.config.base_output_dir or not self.config.exam_name:
            return
            
        import json
        from datetime import datetime
        
        experiment_info = {
            "experiment_name": self.config.exam_name,
            "created_at": datetime.now().isoformat(),
            "configuration": {
                "dataset_names": self.config.dataset_names,
                "base_data_dir": self.config.base_data_dir,
                "base_output_dir": self.config.base_output_dir,
                "extractor_model": self.config.extractor_config.model_name,
                "captioner_model": self.config.captioner_config.model_name,
                "llm_handler_model": self.config.llm_handler_config.pretrained,
                "eval_method": self.config.evaluator_config.eval_method
            }
        }
        
        info_file = Path(self.config.base_output_dir) / f"{self.config.exam_name}_experiment_info.json"
        
        try:
            with open(info_file, 'w', encoding='utf-8') as f:
                json.dump(experiment_info, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Created experiment info file: {info_file}")
        except Exception as e:
            self.logger.warning(f"Failed to create experiment info file: {e}")
    
    def _create_output_directories(self):
        """创建必要的输出目录"""
        # 为每个数据集创建目录
        for dataset_name in self.config.dataset_names:
            directories = [
                f"{self.config.base_data_dir}/{dataset_name}/frames",
                f"{self.config.base_data_dir}/{dataset_name}/captions",
                f"{self.config.base_data_dir}/{dataset_name}/features/visual",
                f"{self.config.base_data_dir}/{dataset_name}/features/text",
            ]
            
            for dir_path in directories:
                Path(dir_path).mkdir(parents=True, exist_ok=True)
                self.logger.info(f"Created directory: {dir_path}")
        
        # 公共输出目录
        common_directories = [
            self.config.dataset_builder_config.save_dir,
            self.config.llm_query_config.result_dir,
            self.config.llm_query_config.scores_dir,
            self.config.util_config.similarity_scores_dir,
            self.config.evaluator_config.output_dir
        ]
        
        for dir_path in common_directories:
            if dir_path:
                Path(dir_path).mkdir(parents=True, exist_ok=True)
                self.logger.info(f"Created directory: {dir_path}")
    
    def extract_frames(self):
        """步骤1: 从视频中提取帧"""
        self.logger.info("Starting frame extraction...")
        
        # 创建独立的帧提取器实例（不加载BLIP-2模型）
        frame_extractor = FrameExtractor()
        
        # 为每个数据集提取帧
        for dataset_name in self.config.dataset_names:
            self.logger.info(f"Extracting frames for dataset: {dataset_name}")
            
            # 设置当前数据集的路径
            videos_dir = f"{self.config.base_data_dir}/{dataset_name}/videos"
            frames_dir = f"{self.config.base_data_dir}/{dataset_name}/frames"
            annotations_file = f"{self.config.base_data_dir}/{dataset_name}/annotations/test.txt"
            
            try:
                # 执行帧提取
                frame_extractor.extract_frames_from_directory(
                    videos_dir=videos_dir,
                    frames_dir=frames_dir,
                    annotations_file=annotations_file,
                    skip_if_exists=True
                )
                self.logger.info(f"Frame extraction completed for {dataset_name}")
            except Exception as e:
                self.logger.error(f"Error extracting frames for {dataset_name}: {e}")
                raise
        
        self.logger.info("Frame extraction completed for all datasets")
    
    def build_dataset(self):
        """步骤2: 构建数据集"""
        self.logger.info("Building dataset...")
        
        # 创建DatasetBuilder实例
        dataset_builder = DatasetBuilder(self.config.dataset_builder_config)
        
        try:
            # 执行数据集构建
            dataset_builder.build()
            self.logger.info("Dataset building completed successfully")
        finally:
            # 清理内存
            del dataset_builder
            gc.collect()
    
    def _check_captions_exist(self, dataset_name):
        """检查字幕是否已存在"""
        captions_dir = Path(f"{self.config.base_data_dir}/{dataset_name}/captions")
        videos_dir = Path(f"{self.config.base_data_dir}/{dataset_name}/videos")
        
        if not captions_dir.exists():
            return False, []
            
        # 检查视频文件夹中的视频文件
        video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v')
        video_files = [f for f in videos_dir.iterdir() if f.suffix.lower() in video_extensions]
        
        # 检查对应的字幕文件是否存在
        missing_captions = []
        for video_file in video_files:
            caption_file = captions_dir / f"{video_file.stem}.txt"
            if not caption_file.exists():
                missing_captions.append(video_file.name)
        
        all_exist = len(missing_captions) == 0
        return all_exist, missing_captions

    def get_video_captions(self):
        """步骤3: 获取视频字幕"""
        self.logger.info("Getting video captions...")
        
        # 首先检查所有数据集是否都已有字幕
        all_datasets_complete = True
        datasets_to_process = []
        
        for dataset_name in self.config.dataset_names:
            captions_exist, missing_files = self._check_captions_exist(dataset_name)
            if captions_exist:
                self.logger.info(f"Captions for dataset {dataset_name} already exist, skipping")
            else:
                all_datasets_complete = False
                datasets_to_process.append((dataset_name, missing_files))
                self.logger.info(f"Dataset {dataset_name} needs caption generation for {len(missing_files)} videos")
        
        if all_datasets_complete:
            self.logger.info("All caption files already exist, skipping caption generation")
            return
        
        # 只有在需要处理时才加载模型
        self.logger.info("Loading caption model...")
        captioner = None
        
        try:
            # 为需要处理的数据集生成字幕
            for dataset_name, missing_files in datasets_to_process:
                self.logger.info(f"Getting captions for dataset: {dataset_name}")
                
                # 创建针对当前数据集的配置
                dataset_captioner_config = self.config.captioner_config
                dataset_captioner_config.video_folder = f"{self.config.base_data_dir}/{dataset_name}/videos"
                dataset_captioner_config.output_folder = f"{self.config.base_data_dir}/{dataset_name}/captions"
                
                # 只在第一次需要时创建并加载模型
                if captioner is None:
                    captioner = VideoCaptioner(dataset_captioner_config)
                    captioner.load_model()
                else:
                    # 更新配置路径
                    captioner.config = dataset_captioner_config
                
                # 执行字幕生成
                captioner.caption_video_folder()
                self.logger.info(f"Video captioning completed for {dataset_name}")
                
        finally:
            # 清理内存
            if captioner is not None:
                del captioner
            gc.collect()
        
        self.logger.info("Video captioning completed for all datasets")
    
    def _check_visual_features_exist(self, dataset_name):
        """检查视觉特征是否已存在"""
        features_dir = Path(f"{self.config.base_data_dir}/{dataset_name}/features/visual")
        videos_dir = Path(f"{self.config.base_data_dir}/{dataset_name}/videos")
        
        if not features_dir.exists():
            return False, []
            
        # 检查视频文件夹中的视频文件
        video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v')
        video_files = [f for f in videos_dir.iterdir() if f.suffix.lower() in video_extensions]
        
        # 检查对应的特征文件是否存在
        missing_features = []
        for video_file in video_files:
            feature_file = features_dir / f"{video_file.stem}.npy"
            if not feature_file.exists():
                missing_features.append(video_file.name)
        
        all_exist = len(missing_features) == 0
        return all_exist, missing_features

    def extract_visual_features(self):
        """步骤4: 提取视觉特征"""
        self.logger.info("Extracting visual features...")
        
        # 首先检查所有数据集是否都已有视觉特征
        all_datasets_complete = True
        datasets_to_process = []
        
        for dataset_name in self.config.dataset_names:
            features_exist, missing_files = self._check_visual_features_exist(dataset_name)
            if features_exist:
                self.logger.info(f"Visual features for dataset {dataset_name} already exist, skipping")
            else:
                all_datasets_complete = False
                datasets_to_process.append((dataset_name, missing_files))
                self.logger.info(f"Dataset {dataset_name} needs visual feature extraction for {len(missing_files)} videos")
        
        if all_datasets_complete:
            self.logger.info("All visual feature files already exist, skipping visual feature extraction")
            return
        
        # 只有在需要处理时才加载模型
        self.logger.info("Loading visual feature extraction model...")
        extractor = None
        
        try:
            # 为需要处理的数据集提取视觉特征
            for dataset_name, missing_files in datasets_to_process:
                self.logger.info(f"Extracting visual features for dataset: {dataset_name}")
                
                # 只在第一次需要时创建模型
                if extractor is None:
                    extractor = Extractor(self.config.extractor_config)
                
                # 执行视觉特征提取
                extractor.extract_visual_features(
                    video_folder=f"{self.config.base_data_dir}/{dataset_name}/videos",
                    frames_root_dir=f"{self.config.base_data_dir}/{dataset_name}/frames",
                    output_folder=f"{self.config.base_data_dir}/{dataset_name}/features/visual",
                    stride=self.config.extractor_config.stride,
                    batch_size=self.config.extractor_config.batch_size
                )
                self.logger.info(f"Visual feature extraction completed for {dataset_name}")
                
        finally:
            # 清理内存
            if extractor is not None:
                del extractor
            gc.collect()
        
        self.logger.info("Visual feature extraction completed for all datasets")
    
    def _check_text_features_exist(self, dataset_name):
        """检查文本特征是否已存在"""
        features_dir = Path(f"{self.config.base_data_dir}/{dataset_name}/features/text")
        captions_dir = Path(f"{self.config.base_data_dir}/{dataset_name}/captions")
        
        if not features_dir.exists():
            return False, []
            
        # 检查字幕文件夹中的文本文件
        caption_files = [f for f in captions_dir.iterdir() if f.suffix.lower() == '.txt']
        
        # 检查对应的特征文件是否存在
        missing_features = []
        for caption_file in caption_files:
            feature_file = features_dir / f"{caption_file.stem}.npy"
            if not feature_file.exists():
                missing_features.append(caption_file.name)
        
        all_exist = len(missing_features) == 0
        return all_exist, missing_features

    def extract_text_features(self):
        """步骤5: 提取文本特征"""
        self.logger.info("Extracting text features...")
        
        # 首先检查所有数据集是否都已有文本特征
        all_datasets_complete = True
        datasets_to_process = []
        
        for dataset_name in self.config.dataset_names:
            features_exist, missing_files = self._check_text_features_exist(dataset_name)
            if features_exist:
                self.logger.info(f"Text features for dataset {dataset_name} already exist, skipping")
            else:
                all_datasets_complete = False
                datasets_to_process.append((dataset_name, missing_files))
                self.logger.info(f"Dataset {dataset_name} needs text feature extraction for {len(missing_files)} caption files")
        
        if all_datasets_complete:
            self.logger.info("All text feature files already exist, skipping text feature extraction")
            return
        
        # 只有在需要处理时才加载模型
        self.logger.info("Loading text feature extraction model...")
        extractor = None
        
        try:
            # 为需要处理的数据集提取文本特征
            for dataset_name, missing_files in datasets_to_process:
                self.logger.info(f"Extracting text features for dataset: {dataset_name}")
                
                # 只在第一次需要时创建模型
                if extractor is None:
                    extractor = Extractor(self.config.extractor_config)
                
                # 执行文本特征提取
                extractor.extract_text_features(
                    text_folder=f"{self.config.base_data_dir}/{dataset_name}/captions",
                    output_folder=f"{self.config.base_data_dir}/{dataset_name}/features/text"
                )
                self.logger.info(f"Text feature extraction completed for {dataset_name}")
                
        finally:
            # 清理内存
            if extractor is not None:
                del extractor
            gc.collect()
        
        self.logger.info("Text feature extraction completed for all datasets")
    
    def query_llm(self):
        """步骤6: 查询LLM"""
        self.logger.info("Querying LLM...")
        
        # 检查数据集是否存在
        dataset_dir = Path(self.config.llm_query_config.dataset_dir)
        if not dataset_dir.exists():
            self.logger.warning(f"Dataset directory {dataset_dir} does not exist, skipping LLM query")
            return
        
        # 只有在需要时才加载LLM模型
        self.logger.info("Loading LLM model for querying...")
        llm_handler = None
        llm_query = None
        
        try:
            # 创建LLMHandler实例
            llm_handler = LLMHandler(self.config.llm_handler_config)
            
            # 创建LLMQuery实例
            llm_query = LLMQuery(self.config.llm_query_config, llm_handler)
            
            # 执行LLM查询流程
            llm_query.run_query_pipeline(
                dataset_dir=Path(self.config.llm_query_config.dataset_dir),
                result_dir=Path(self.config.llm_query_config.result_dir),
                scores_dir=Path(self.config.llm_query_config.scores_dir),
                frame_interval=self.config.llm_query_config.frame_interval
            )
            self.logger.info("LLM querying completed successfully")
        finally:
            # 清理内存
            if llm_handler is not None:
                del llm_handler
            if llm_query is not None:
                del llm_query
            gc.collect()
    
    def calc_semantic_similarity(self):
        """步骤7: 计算语义相似度"""
        self.logger.info("Calculating semantic similarity...")
        
        # 为每个数据集计算语义相似度
        for dataset_name in self.config.dataset_names:
            self.logger.info(f"Calculating semantic similarity for dataset: {dataset_name}")
            
            # 创建针对当前数据集的配置
            dataset_util_config = self.config.util_config
            dataset_util_config.visual_features_dir = f"{self.config.base_data_dir}/{dataset_name}/features/visual"
            dataset_util_config.text_features_dir = f"{self.config.base_data_dir}/{dataset_name}/features/text"
            # similarity_scores_dir 保持为公共目录，但在函数内部应该按数据集分别处理
            
            try:
                # 使用util模块计算相似度，传入配置对象
                process_all(dataset_util_config)
                self.logger.info(f"Semantic similarity calculation completed for {dataset_name}")
            finally:
                # 强制垃圾回收
                gc.collect()
        
        self.logger.info("Semantic similarity calculation completed for all datasets")
    
    def eval_final_score(self):
        """步骤8: 评估最终分数"""
        self.logger.info("Evaluating final scores...")
        
        # 创建VideoSummaryEvaluator实例
        evaluator = VideoSummaryEvaluator(self.config.evaluator_config)
        
        try:
            # 执行评估
            results = evaluator.evaluate_from_file(self.config.evaluator_config.llm_score_file)
            
            self.logger.info("Final score evaluation completed successfully")
            self.logger.info(f"Evaluation results: {results}")
            
            return results
        finally:
            # 清理内存
            del evaluator
            gc.collect()
    
    def run(self):
        """主入口点：运行完整的视频摘要生成流程"""
        self.logger.info("Starting Video Summarization Pipeline...")
        
        if self.config.exam_name:
            self.logger.info(f"Running experiment: {self.config.exam_name}")
        
        try:
            # 创建输出目录
            self._create_output_directories()
            
            # 按顺序执行流程步骤
            self.extract_frames()
            self.build_dataset()
            self.get_video_captions()
            self.extract_visual_features()
            self.extract_text_features()
            self.query_llm()
            self.calc_semantic_similarity()
            results = self.eval_final_score()
            
            # 保存最终结果到实验目录
            self._save_final_results(results)
            
            self.logger.info("Video Summarization Pipeline completed successfully!")
            if self.config.exam_name:
                self.logger.info(f"Experiment '{self.config.exam_name}' completed successfully!")
                self.logger.info(f"All results saved to: {self.config.base_output_dir}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            if self.config.exam_name:
                self.logger.error(f"Experiment '{self.config.exam_name}' failed!")
            raise
        finally:
            # 最终清理
            gc.collect()
    
    def _save_final_results(self, results):
        """保存最终结果到实验目录"""
        if not self.config.base_output_dir or not self.config.exam_name:
            return
            
        import json
        from datetime import datetime
        
        final_results = {
            "experiment_name": self.config.exam_name,
            "completed_at": datetime.now().isoformat(),
            "results": results,
            "status": "completed"
        }
        
        results_file = Path(self.config.base_output_dir) / f"{self.config.exam_name}_final_results.json"
        
        try:
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(final_results, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Saved final results to: {results_file}")
        except Exception as e:
            self.logger.warning(f"Failed to save final results: {e}")


# 示例使用
if __name__ == "__main__":
    # 创建配置
    config = VideoSummarizationPipelineConfig(
        base_data_dir="/root/autodl-tmp/data",
        base_output_dir="/root/TFVSN",
        dataset_names=["TVSum", "SumMe"]  # 修改为多个数据集
    )
    
    # 创建并运行流程
    pipeline = VideoSummarizationPipeline(config)
    results = pipeline.run()
    print("Pipeline results:", results)
