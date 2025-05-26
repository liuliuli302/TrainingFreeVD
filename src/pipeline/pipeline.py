"""
视频摘要生成Pipeline
"""
import gc
import logging
from pathlib import Path
from config.config import VideoSummarizationPipelineConfig
from llm.extractor import Extractor
from dataset.dataset_builder import DatasetBuilder
from llm.captioner import VideoCaptioner
from llm.llm_handler import LLMHandler
from llm.llm_query import LLMQuery
from eval.evaluator import VideoSummaryEvaluator
from util.util import process_all


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
                "dataset_name": self.config.dataset_name,
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
        directories = [
            self.config.extractor_config.frames_dir,
            self.config.dataset_builder_config.save_dir,
            self.config.captioner_config.output_folder,
            self.config.llm_query_config.result_dir,
            self.config.llm_query_config.scores_dir,
            self.config.util_config.visual_features_dir,
            self.config.util_config.text_features_dir,
            self.config.util_config.similarity_scores_dir,
            self.config.evaluator_config.output_dir
        ]
        
        for dir_path in directories:
            if dir_path:
                Path(dir_path).mkdir(parents=True, exist_ok=True)
                self.logger.info(f"Created directory: {dir_path}")
    
    def extract_frames(self):
        """步骤1: 从视频中提取帧"""
        self.logger.info("Starting frame extraction...")
        
        # 创建Extractor实例
        extractor = Extractor(self.config.extractor_config)
        
        try:
            # 执行帧提取
            extractor.extract_frames_from_directory()
            self.logger.info("Frame extraction completed successfully")
        finally:
            # 清理内存
            del extractor
            gc.collect()
    
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
    
    def get_video_captions(self):
        """步骤3: 获取视频字幕"""
        self.logger.info("Getting video captions...")
        
        # 创建VideoCaptioner实例
        captioner = VideoCaptioner(self.config.captioner_config)
        
        try:
            # 加载模型
            captioner.load_model()
            
            # 执行字幕生成
            captioner.caption_video_folder()
            self.logger.info("Video captioning completed successfully")
        finally:
            # 清理内存
            del captioner
            gc.collect()
    
    def extract_visual_features(self):
        """步骤4: 提取视觉特征"""
        self.logger.info("Extracting visual features...")
        
        # 创建Extractor实例
        extractor = Extractor(self.config.extractor_config)
        
        try:
            # 执行视觉特征提取
            extractor.extract_visual_features(
                video_folder=self.config.extractor_config.videos_dir,
                output_folder=self.config.util_config.visual_features_dir,
                stride=self.config.extractor_config.stride,
                batch_size=self.config.extractor_config.batch_size
            )
            self.logger.info("Visual feature extraction completed successfully")
        finally:
            # 清理内存
            del extractor
            gc.collect()
    
    def extract_text_features(self):
        """步骤5: 提取文本特征"""
        self.logger.info("Extracting text features...")
        
        # 创建Extractor实例
        extractor = Extractor(self.config.extractor_config)
        
        try:
            # 执行文本特征提取
            extractor.extract_text_features(
                text_folder=self.config.captioner_config.output_folder,
                output_folder=self.config.util_config.text_features_dir
            )
            self.logger.info("Text feature extraction completed successfully")
        finally:
            # 清理内存
            del extractor
            gc.collect()
    
    def query_llm(self):
        """步骤6: 查询LLM"""
        self.logger.info("Querying LLM...")
        
        # 创建LLMHandler实例
        llm_handler = LLMHandler(self.config.llm_handler_config)
        
        # 创建LLMQuery实例
        llm_query = LLMQuery(self.config.llm_query_config, llm_handler)
        
        try:
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
            del llm_handler
            del llm_query
            gc.collect()
    
    def calc_semantic_similarity(self):
        """步骤7: 计算语义相似度"""
        self.logger.info("Calculating semantic similarity...")
        
        try:
            # 使用util模块计算相似度，传入配置对象
            process_all(self.config.util_config)
            self.logger.info("Semantic similarity calculation completed successfully")
        finally:
            # 强制垃圾回收
            gc.collect()
    
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
        dataset_name="TVSum"
    )
    
    # 创建并运行流程
    pipeline = VideoSummarizationPipeline(config)
    results = pipeline.run()
    print("Pipeline results:", results)
