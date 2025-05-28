import argparse
from pathlib import Path
import json
import numpy as np
import copy
import os
import re
from tqdm import tqdm
from typing import Dict, List, Any, TYPE_CHECKING

from src.config.config import LLMQueryConfig
from .llm_handler import LLMHandler
import h5py


class LLMQuery:
    """Query LLM for video summarization scores"""

    def __init__(self, config: LLMQueryConfig, llm_handler: LLMHandler):
        self.config = config
        self.llm_handler = llm_handler
        # 使用配置中的常量
        self.score_pattern = config.score_pattern

        # 设计多个专门针对视频摘要的prompts
        self.video_summary_prompts = [
            # Prompt 1: 基于重要性和代表性
            {
                "name": "importance_representativeness",
                "prompt": """You are a professional video editor. Please analyze these video frames and rate each frame's importance for video summarization.

Consider these factors for each frame:
1. Visual content significance and uniqueness
2. How well it represents the main events or actions
3. Information density and clarity
4. Temporal importance in the video narrative

Rate each frame on a scale from 0.0 to 1.0, where:
- 0.0 = Not important, redundant, or unclear
- 1.0 = Highly important, unique, and representative

Please provide ONLY a score list in the format: [0.1, 0.8, 0.3, 0.9, ...]
No additional text or explanation."""
            },

            # Prompt 2: 基于视觉多样性和信息量
            {
                "name": "diversity_information",
                "prompt": """You are an AI video analysis expert. Evaluate these video frames for creating an effective video summary.

For each frame, consider:
1. Visual diversity compared to other frames
2. Amount of visual information and detail
3. Scene transition significance
4. Object/person prominence and clarity

Assign scores from 0.0 to 1.0:
- 0.0 = Low diversity, minimal information, similar to other frames
- 1.0 = High diversity, rich information, distinct from other frames

Output format: [score1, score2, score3, ...]
Only provide the numerical scores, no explanations."""
            },

            # Prompt 3: 基于关键事件和动作
            {
                "name": "key_events_actions",
                "prompt": """As a video content analyzer, rate these frames based on their relevance to key events and actions.

Evaluate each frame for:
1. Presence of significant actions or events
2. Critical moments in the video timeline
3. Change points or transitions
4. Human activities and interactions

Score from 0.0 to 1.0:
- 0.0 = Static, no significant events, transition periods
- 1.0 = Major events, key actions, important moments

Return only: [0.x, 0.x, 0.x, ...]"""
            },

            # Prompt 4: 基于视觉质量和清晰度
            {
                "name": "quality_clarity",
                "prompt": """You are a video quality assessment expert. Rate these frames for summary inclusion based on visual quality.

Consider for each frame:
1. Image sharpness and clarity
2. Lighting and exposure quality
3. Composition and framing
4. Overall visual appeal and readability

Rate from 0.0 to 1.0:
- 0.0 = Poor quality, blurry, poorly lit, bad composition
- 1.0 = Excellent quality, sharp, well-lit, good composition

Format: [score1, score2, score3, ...]
Provide only numerical scores."""
            },

            # Prompt 5: 综合评估
            {
                "name": "comprehensive_summary",
                "prompt": """You are an expert video summarization system. Provide comprehensive scores for these frames.

Rate each frame considering ALL aspects:
1. Content importance and uniqueness
2. Visual quality and clarity
3. Representativeness of video content
4. Temporal significance
5. Information density

Provide balanced scores from 0.0 to 1.0 where:
- 0.0 = Should not be included in summary
- 1.0 = Essential for video summary

Output: [score1, score2, score3, ...]
Only provide the score array."""
            }
        ]

        # 更新的score pattern来匹配我们的输出格式
        self.score_patterns = [
            r'\[([\d\.\,\s]+)\]',  # 匹配 [0.1, 0.2, 0.3, ...]
            r'(\d+\.\d+)',  # 匹配单独的小数
            r'(\d+)',  # 匹配整数
        ]

    def load_datasets(self, dataset_dir: Path) -> List[Path]:
        """Load dataset JSON files"""
        # 使用配置中的数据集文件名
        json_files = [
            # summe_dataset_jump.json
            dataset_dir / "SumMe" / self.config.dataset_file_names[0],
            # summe_dataset_turn.json
            dataset_dir / "SumMe" / self.config.dataset_file_names[1],
            # tvsum_dataset_jump.json
            dataset_dir / "TVSum" / self.config.dataset_file_names[2],
            # tvsum_dataset_turn.json
            dataset_dir / "TVSum" / self.config.dataset_file_names[3],
        ]

        # Check if files exist
        for file in json_files:
            if not file.exists():
                print(f"Warning: {file} does not exist")

        return [f for f in json_files if f.exists()]

    def hdf5_to_dict(self, hdf5_file: Path) -> Dict:
        """Convert HDF5 file to dictionary"""
        def recursively_convert_to_dict(h5_obj):
            if isinstance(h5_obj, h5py.Dataset):
                return h5_obj[()]
            elif isinstance(h5_obj, h5py.Group):
                return {
                    key: recursively_convert_to_dict(h5_obj[key]) for key in h5_obj.keys()
                }
            else:
                raise TypeError(f"Unsupported type: {type(h5_obj)}")

        with h5py.File(hdf5_file, "r") as f:
            return recursively_convert_to_dict(f)

    def initialize_score_list(self, dataset_dir: Path) -> Dict:
        """Initialize score list structure from HDF5 files"""
        # 使用配置中的HDF5文件名
        summe_hdf = dataset_dir / "SumMe" / \
            self.config.hdf5_file_names["summe"]
        tvsum_hdf = dataset_dir / "TVSum" / \
            self.config.hdf5_file_names["tvsum"]

        summe_dict = self.hdf5_to_dict(summe_hdf)
        tvsum_dict = self.hdf5_to_dict(tvsum_hdf)

        data_dict_list = {"summe": summe_dict, "tvsum": tvsum_dict}

        # Initialize score list
        score_list = {"summe": {}, "tvsum": {}}
        for dataset_name in data_dict_list.keys():
            for video_name in data_dict_list[dataset_name].keys():
                data_dict = data_dict_list[dataset_name]
                scores = np.zeros(int(data_dict[video_name]["n_steps"]))
                score_list[dataset_name][video_name] = scores

        return score_list

    def get_score_index_from_image_list(self, image_list: List[str], frame_interval: int = 15) -> List[int]:
        """Get score indices from image list"""
        score_index = []
        for path in image_list:
            index = int(int(os.path.basename(
                path).split(".")[0]) / frame_interval)
            score_index.append(index)
        return score_index

    def extract_scores_from_response(self, llm_response: str) -> List[float]:
        """Extract scores from LLM response with multiple patterns"""
        scores = []

        # Try different patterns to extract scores
        for pattern in self.score_patterns:
            matches = re.findall(pattern, llm_response)
            if matches:
                if pattern == r'\[([\d\.\,\s]+)\]':  # Array format
                    # Parse array content
                    array_content = matches[0]
                    score_strings = re.findall(r'\d+\.\d+|\d+', array_content)
                    scores = [float(s) for s in score_strings]
                    break
                else:  # Individual numbers
                    scores = [float(match) for match in matches]
                    break

        return scores

    def query_single_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Query LLM for a single sample using multiple prompts"""
        try:
            # Preprocess images
            images = self.llm_handler.preprocess_images(sample["images"])
            video_time = sample["video_time"]
            frame_time = sample["frame_time"]

            # 初始化结果样本
            result_sample = copy.deepcopy(sample)
            result_sample["llm_responses"] = {}
            result_sample["extracted_scores"] = {}

            # 对每个prompt进行单次查询
            for prompt_info in self.video_summary_prompts:
                prompt_name = prompt_info["name"]
                prompt_text = prompt_info["prompt"]

                try:
                    # 使用单次生成而不是多轮对话
                    response = self.llm_handler.generate_response(
                        images=images,
                        prompt=prompt_text,
                        video_time=video_time,
                        frame_time=frame_time,
                        max_new_tokens=500,  # 减少token数量，因为只需要分数列表
                        temperature=0.0
                    )

                    # 保存原始响应
                    result_sample["llm_responses"][prompt_name] = response

                    # 提取分数
                    scores = self.extract_scores_from_response(response)
                    result_sample["extracted_scores"][prompt_name] = scores

                except Exception as e:
                    from tqdm import tqdm
                    tqdm.write(f"    Error with prompt {prompt_name}: {str(e)}")
                    result_sample["llm_responses"][prompt_name] = f"Error: {str(e)}"
                    result_sample["extracted_scores"][prompt_name] = []

            # 为了保持向后兼容，选择一个主要的输出
            # 使用综合评估作为主要输出
            if "comprehensive_summary" in result_sample["llm_responses"]:
                result_sample["llm_out"] = result_sample["llm_responses"]["comprehensive_summary"]
            else:
                # 如果综合评估失败，使用第一个成功的响应
                for prompt_info in self.video_summary_prompts:
                    prompt_name = prompt_info["name"]
                    if prompt_name in result_sample["llm_responses"] and not result_sample["llm_responses"][prompt_name].startswith("Error:"):
                        result_sample["llm_out"] = result_sample["llm_responses"][prompt_name]
                        break
                else:
                    result_sample["llm_out"] = "All prompts failed"

            return result_sample

        except Exception as e:
            error_msg = str(e)
            sample_id = sample.get('id', 'unknown')

            if "meta" in error_msg.lower() or "cannot copy" in error_msg.lower():
                print(
                    f"Meta tensor error processing sample {sample_id}: {error_msg}")
                print(
                    "This indicates a model loading issue. Consider restarting the pipeline or using a different device_map.")
            else:
                print(f"Error processing sample {sample_id}: {error_msg}")

            result_sample = copy.deepcopy(sample)
            result_sample["llm_out"] = f"Error: {error_msg}"
            result_sample["llm_responses"] = {"error": f"Error: {error_msg}"}
            result_sample["extracted_scores"] = {}
            return result_sample

    def process_dataset_file(self, dataset_file: Path, result_dir: Path) -> Path:
        """Process a single dataset file with multiple prompts"""
        result_file_name = str(dataset_file.name).split('.')[
            0] + "_result.json"
        result_file = result_dir / result_file_name

        # 为每个prompt创建单独的结果文件
        prompt_result_files = []
        for prompt_info in self.video_summary_prompts:
            prompt_name = prompt_info["name"]
            prompt_file_name = str(dataset_file.name).split('.')[
                0] + f"_{prompt_name}_result.json"
            prompt_result_files.append(
                (prompt_name, result_dir / prompt_file_name))

        # Load dataset
        with open(dataset_file, "r") as f:
            dataset = json.load(f)

        results = []
        prompt_results = {name: [] for name, _ in prompt_result_files}

        print(f"Processing {dataset_file.name}...")
        print(f"Total samples: {len(dataset)}")
        print(f"Using {len(self.video_summary_prompts)} different prompts")

        for sample in tqdm(dataset, desc="Samples"):
            result_sample = self.query_single_sample(sample)
            results.append(result_sample)

            # 为每个prompt保存单独的结果
            for prompt_info in self.video_summary_prompts:
                prompt_name = prompt_info["name"]
                if prompt_name in result_sample.get("llm_responses", {}):
                    prompt_specific_sample = copy.deepcopy(sample)
                    prompt_specific_sample["llm_out"] = result_sample["llm_responses"][prompt_name]
                    prompt_specific_sample["prompt_name"] = prompt_name
                    prompt_specific_sample["prompt_text"] = prompt_info["prompt"]
                    if prompt_name in result_sample.get("extracted_scores", {}):
                        prompt_specific_sample["extracted_scores"] = result_sample["extracted_scores"][prompt_name]
                    prompt_results[prompt_name].append(prompt_specific_sample)

        # 保存综合结果
        with open(result_file, "w") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        print(f"Comprehensive results saved to {result_file}")

        # 保存每个prompt的单独结果
        for prompt_name, prompt_file in prompt_result_files:
            if prompt_results[prompt_name]:  # 只保存非空结果
                with open(prompt_file, "w") as f:
                    json.dump(prompt_results[prompt_name],
                              f, indent=4, ensure_ascii=False)
                print(f"Prompt '{prompt_name}' results saved to {prompt_file}")

        return result_file

    def process_results_to_scores(self, result_files: List[Path],
                                  score_list: Dict, frame_interval: int = 15) -> Dict:
        """Process LLM results to extract scores from multiple prompts"""
        result_dict = {}

        for file in result_files:
            with open(file, "r") as f:
                result = json.load(f)

            file_name = file.stem
            dataset_name = file_name.split("_")[0]

            if dataset_name not in score_list:
                print(
                    f"Warning: Dataset {dataset_name} not found in score_list")
                continue

            # 为每个prompt创建单独的分数字典
            prompt_score_dicts = {}
            for prompt_info in self.video_summary_prompts:
                prompt_name = prompt_info["name"]
                prompt_score_dicts[prompt_name] = copy.deepcopy(
                    score_list[dataset_name])

            # 主要分数字典（用于向后兼容）
            scores_list_tmp = copy.deepcopy(score_list[dataset_name])
            check_out = []

            for sample in result:
                try:
                    # Extract video name
                    video_name = "_".join(sample["id"].split("_")[1:3])

                    if video_name not in scores_list_tmp:
                        print(
                            f"Warning: Video {video_name} not found in score list")
                        continue

                    # Get indices
                    images = sample["images"]
                    index = self.get_score_index_from_image_list(
                        images, frame_interval)
                    index = np.array(index)

                    # 处理每个prompt的分数
                    if "extracted_scores" in sample:
                        for prompt_name, scores in sample["extracted_scores"].items():
                            if prompt_name in prompt_score_dicts and scores:
                                scores_array = np.array(scores)

                                # 调整分数长度
                                if len(scores_array) > len(index):
                                    scores_array = scores_array[:len(index)]
                                elif len(scores_array) < len(index):
                                    scores_array = np.concatenate(
                                        (scores_array, np.zeros(
                                            len(index) - len(scores_array)))
                                    )

                                # 更新prompt特定的分数
                                prompt_score_dicts[prompt_name][video_name][index] = scores_array

                    # 处理主要输出（向后兼容）
                    llm_out = sample["llm_out"]
                    scores = self.extract_scores_from_response(llm_out)

                    if scores:
                        scores = np.array(scores)

                        # 调整分数长度
                        if len(scores) == len(index):
                            check_out.append(1)
                        else:
                            if len(scores) < len(index):
                                scores = np.concatenate(
                                    (scores, np.zeros(len(index) - len(scores)))
                                )
                            elif len(scores) > len(index):
                                scores = scores[:len(index)]
                            check_out.append(0)

                        # 更新主要分数
                        scores_list_tmp[video_name][index] = scores
                    else:
                        check_out.append(0)

                except Exception as e:
                    print(
                        f"Error processing sample {sample.get('id', 'unknown')}: {str(e)}")
                    check_out.append(0)

            # 转换为列表格式用于JSON序列化
            for video_name in scores_list_tmp:
                scores_list_tmp[video_name] = scores_list_tmp[video_name].tolist()

            for prompt_name in prompt_score_dicts:
                for video_name in prompt_score_dicts[prompt_name]:
                    prompt_score_dicts[prompt_name][video_name] = prompt_score_dicts[prompt_name][video_name].tolist(
                    )

            # 打印验证统计
            check_out = np.array(check_out)
            right = check_out.sum()
            length = len(check_out)
            print(f"{file_name} - Score extraction rate: {right}/{length}")

            # 保存结果
            result_dict[file_name] = {
                "main_scores": scores_list_tmp,
                "prompt_scores": prompt_score_dicts,
                "extraction_rate": f"{right}/{length}"
            }

        return result_dict

    def run_query_pipeline(self, dataset_dir=None, result_dir=None, scores_dir=None, frame_interval=None):
        """
        Run the complete query pipeline

        Args:
            dataset_dir: Dataset directory（可选，默认使用config中的参数）
            result_dir: Result directory（可选，默认使用config中的参数）
            scores_dir: Scores directory（可选，默认使用config中的参数）
            frame_interval: Frame interval（可选，默认使用config中的参数）
        """
        dataset_dir = dataset_dir or Path(self.config.dataset_dir)
        result_dir = result_dir or Path(self.config.result_dir)
        scores_dir = scores_dir or Path(self.config.scores_dir)
        frame_interval = frame_interval or self.config.frame_interval
        """Run the complete query pipeline"""
        # Create output directories
        result_dir.mkdir(parents=True, exist_ok=True)
        scores_dir.mkdir(parents=True, exist_ok=True)

        # Load datasets
        dataset_files = self.load_datasets(dataset_dir)

        # Process each dataset file
        result_files = []
        for dataset_file in dataset_files:
            result_file = self.process_dataset_file(dataset_file, result_dir)
            result_files.append(result_file)

        # Initialize score list
        score_list = self.initialize_score_list(dataset_dir)

        # Process results to extract scores
        result_dict = self.process_results_to_scores(
            result_files, score_list, frame_interval)

        # Save final scores
        scores_json = scores_dir / "raw_llm_out_scores.json"
        with open(scores_json, "w") as f:
            json.dump(result_dict, f, separators=(",", ": "), indent=4)

        print(f"Final scores saved to {scores_json}")


def main():
    parser = argparse.ArgumentParser(
        description="Query LLM for video summarization")
    parser.add_argument("--dataset_dir", type=str, required=True,
                        help="Directory containing datasets")
    parser.add_argument("--result_dir", type=str, required=True,
                        help="Directory to save LLM query results")
    parser.add_argument("--scores_dir", type=str, required=True,
                        help="Directory to save extracted scores")
    parser.add_argument("--model_type", type=str, default="local",
                        choices=["local", "api"], help="Model type: local or api")
    parser.add_argument("--model_name", type=str, default="lmms-lab/LLaVA-Video-7B-Qwen2",
                        help="Model name or path")
    parser.add_argument("--frame_interval", type=int, default=15,
                        help="Frame interval for score indexing")

    args = parser.parse_args()

    # Initialize LLM handler
    if args.model_type == "local":
        llm_handler = LLMHandler(
            model_type="local",
            pretrained=args.model_name
        )
    else:
        llm_handler = LLMHandler(
            model_type="api",
            api_key=os.getenv("OPENAI_API_KEY"),  # Adjust as needed
            base_url=os.getenv("OPENAI_BASE_URL")
        )

    # Initialize query processor
    query_processor = LLMQuery(llm_handler)

    # Run pipeline
    query_processor.run_query_pipeline(
        dataset_dir=Path(args.dataset_dir),
        result_dir=Path(args.result_dir),
        scores_dir=Path(args.scores_dir),
        frame_interval=args.frame_interval
    )


if __name__ == "__main__":
    main()
