import json
import numpy as np
import h5py
from pathlib import Path
from typing import Dict, List, Tuple, Union, TYPE_CHECKING, Any
import os

from src.config.config import EvaluatorConfig


class VideoSummaryEvaluator:
    """Video Summarization Evaluator for SumMe and TVSum datasets."""
    
    def __init__(self, config: 'EvaluatorConfig'):
        """
        Initialize the evaluator with configuration.
        
        Args:
            config: EvaluatorConfig配置对象
        """
        self.config = config
        self.dataset_dir = Path(config.dataset_dir)
        # 使用配置中的HDF5文件名常量
        self.summe_hdf = self.dataset_dir / "SumMe" / config.hdf5_file_names["summe"]
        self.tvsum_hdf = self.dataset_dir / "TVSum" / config.hdf5_file_names["tvsum"]
        
        # Load dataset dictionaries
        self.summe_dict = self._hdf5_to_dict(self.summe_hdf)
        self.tvsum_dict = self._hdf5_to_dict(self.tvsum_hdf)
        self.data_dict_list = {"summe": self.summe_dict, "tvsum": self.tvsum_dict}
    
    def _hdf5_to_dict(self, hdf5_file: Path) -> Dict:
        """Convert HDF5 file to dictionary."""
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
    
    def _knapsack(self, W: int, wt: List[int], val: List[float], n: int) -> List[int]:
        """Knapsack algorithm for shot selection."""
        K = [[0 for _ in range(W + 1)] for _ in range(n + 1)]

        # Build table K[][] in bottom up manner
        for i in range(n + 1):
            for w in range(W + 1):
                if i == 0 or w == 0:
                    K[i][w] = 0
                elif wt[i - 1] <= w:
                    K[i][w] = max(val[i - 1] + K[i - 1][w - wt[i - 1]], K[i - 1][w])
                else:
                    K[i][w] = K[i - 1][w]

        selected = []
        w = W
        for i in range(n, 0, -1):
            if K[i][w] != K[i - 1][w]:
                selected.insert(0, i - 1)
                w -= wt[i - 1]

        return selected
    
    def generate_summary(self, shot_bound: np.ndarray, scores: List[float], 
                        n_frames: int, positions: np.ndarray) -> np.ndarray:
        """Generate summary from frame scores using knapsack algorithm."""
        if positions.dtype != int:
            positions = positions.astype(np.int32)
        if positions[-1] != n_frames:
            positions = np.concatenate([positions, [n_frames]])

        frame_scores = np.zeros(n_frames, dtype=np.float32)
        
        for i in range(len(positions) - 1):
            pos_left, pos_right = positions[i], positions[i + 1]
            if i == len(scores):
                frame_scores[pos_left:pos_right] = 0
            else:
                frame_scores[pos_left:pos_right] = scores[i]

        # Compute shot-level importance scores
        shot_imp_scores = []
        shot_lengths = []
        for shot in shot_bound:
            shot_lengths.append(shot[1] - shot[0] + 1)
            shot_imp_scores.append(
                (frame_scores[shot[0]: shot[1] + 1].mean()).item())

        # Select the best shots using knapsack
        final_shot = shot_bound[-1]
        final_max_length = int((final_shot[1] + 1) * self.config.summary_ratio)

        selected = self._knapsack(
            final_max_length, shot_lengths, shot_imp_scores, len(shot_lengths)
        )

        # Generate summary vector
        summary = np.zeros(final_shot[1] + 1, dtype=np.int8)
        for shot in selected:
            summary[shot_bound[shot][0]: shot_bound[shot][1] + 1] = 1

        return summary
    
    def evaluate_summary(self, predicted_summary: np.ndarray, 
                        user_summary: np.ndarray, eval_method: str = None) -> float:
        """Evaluate predicted summary against user summaries."""
        eval_method = eval_method or self.config.eval_method
        
        max_len = max(len(predicted_summary), user_summary.shape[1])
        S = np.zeros(max_len, dtype=int)
        G = np.zeros(max_len, dtype=int)
        S[: len(predicted_summary)] = predicted_summary

        f_scores = []
        for user in range(user_summary.shape[0]):
            G[: user_summary.shape[1]] = user_summary[user]
            overlapped = S & G

            # Compute precision, recall, f-score
            precision = sum(overlapped) / sum(S) if sum(S) > 0 else 0
            recall = sum(overlapped) / sum(G) if sum(G) > 0 else 0
            if precision + recall == 0:
                f_scores.append(0)
            else:
                f_scores.append(2 * precision * recall * 100 / (precision + recall))

        if eval_method == "max":
            return max(f_scores)
        else:
            return sum(f_scores) / len(f_scores)
    
    def get_fscore_from_predscore(self, pred_scores: List[float], video_name: str, 
                                 dataset_name: str) -> Tuple[float, np.ndarray]:
        """Compute F-score from prediction scores for a single video."""
        dict_video = self.data_dict_list[dataset_name]
        
        shot_bound = dict_video[video_name]["change_points"].astype(int)
        n_frames = dict_video[video_name]["n_frames"]
        positions = dict_video[video_name]["picks"].astype(int)
        user_summary = dict_video[video_name]["user_summary"]
        
        eval_method = "max" if dataset_name == "summe" else "avg"
        
        predicted_summary = self.generate_summary(
            shot_bound, pred_scores, n_frames, positions
        )
        
        fscore = self.evaluate_summary(predicted_summary, user_summary, eval_method)
        
        return fscore, predicted_summary
    
    def evaluate_from_file(self, score_file: str = None) -> Dict[str, float]:
        """
        Evaluate from score file
        
        Args:
            score_file: Path to score file（可选，默认使用config中的参数）
            
        Returns:
            Dictionary with evaluation results
        """
        score_file = score_file or self.config.llm_score_file
        """Evaluate results from a JSON score file."""
        with open(score_file) as f:
            result_list = json.load(f)

        fscore_result = {}
        fscore_result_list = {}
        
        for key in result_list.keys():
            fscore_list = []
            
            # 检查是否为新的多prompt格式
            result_data = result_list[key]
            if isinstance(result_data, dict) and "main_scores" in result_data:
                # 新格式：使用main_scores作为主要分数
                result_scores = result_data["main_scores"]
            else:
                # 旧格式：直接使用分数
                result_scores = result_data
            
            dataset_name = key.split("_")[0]
            
            for video_name in result_scores.keys():
                pred_scores = result_scores[video_name]
                f1score, _ = self.get_fscore_from_predscore(
                    pred_scores, video_name, dataset_name
                )
                fscore_list.append(f1score)
            
            fscore_list = np.array(fscore_list)
            avg_fscore = fscore_list.mean()
            fscore_result[key] = avg_fscore
            fscore_result_list[key] = fscore_list
        
        return fscore_result
    
    def evaluate_single_video(self, pred_scores: List[float], video_name: str, 
                             dataset_name: str) -> float:
        """Evaluate a single video prediction."""
        fscore, _ = self.get_fscore_from_predscore(pred_scores, video_name, dataset_name)
        return fscore
    
    def evaluate_batch(self, predictions: Dict[str, Dict[str, List[float]]]) -> Dict[str, float]:
        """
        Evaluate batch predictions.
        
        Args:
            predictions: Dict with structure {dataset_split: {video_name: scores}}
        
        Returns:
            Dict with average F-scores for each dataset split
        """
        fscore_result = {}
        
        for key, result_scores in predictions.items():
            fscore_list = []
            dataset_name = key.split("_")[0]
            
            for video_name, pred_scores in result_scores.items():
                f1score, _ = self.get_fscore_from_predscore(
                    pred_scores, video_name, dataset_name
                )
                fscore_list.append(f1score)
            
            fscore_list = np.array(fscore_list)
            avg_fscore = fscore_list.mean()
            fscore_result[key] = avg_fscore
        
        return fscore_result
    
    def evaluate_prompt_results(self, score_file: str = None) -> Dict[str, Dict[str, float]]:
        """
        Evaluate results for each individual prompt
        
        Args:
            score_file: Path to score file
            
        Returns:
            Dict with structure {dataset_split: {prompt_name: avg_fscore}}
        """
        score_file = score_file or self.config.llm_score_file
        
        with open(score_file) as f:
            result_list = json.load(f)
        
        prompt_results = {}
        
        for key in result_list.keys():
            result_data = result_list[key]
            
            # 只处理新格式的多prompt结果
            if isinstance(result_data, dict) and "prompt_scores" in result_data:
                dataset_name = key.split("_")[0]
                prompt_results[key] = {}
                
                for prompt_name, prompt_scores in result_data["prompt_scores"].items():
                    fscore_list = []
                    
                    for video_name in prompt_scores.keys():
                        pred_scores = prompt_scores[video_name]
                        f1score, _ = self.get_fscore_from_predscore(
                            pred_scores, video_name, dataset_name
                        )
                        fscore_list.append(f1score)
                    
                    if fscore_list:
                        fscore_list = np.array(fscore_list)
                        avg_fscore = fscore_list.mean()
                        prompt_results[key][prompt_name] = avg_fscore
        
        return prompt_results
    
    def evaluate_comprehensive(self, score_file: str = None) -> Dict[str, Any]:
        """
        Comprehensive evaluation including main scores and per-prompt analysis
        
        Args:
            score_file: Path to score file
            
        Returns:
            Dict with main results and prompt-specific results
        """
        main_results = self.evaluate_from_file(score_file)
        prompt_results = self.evaluate_prompt_results(score_file)
        
        return {
            "main_results": main_results,
            "prompt_results": prompt_results,
            "summary": {
                "overall_avg": np.mean(list(main_results.values())) if main_results else 0.0,
                "prompt_comparison": prompt_results
            }
        }


# Example usage
if __name__ == "__main__":
    # Initialize evaluator
    evaluator = VideoSummaryEvaluator("/path/to/dataset")
    
    # Evaluate from file
    results = evaluator.evaluate_from_file("scores.json")
    print("Results:", results)
    
    # Evaluate single video
    scores = [0.8, 0.6, 0.9, 0.4, 0.7]  # Example scores
    fscore = evaluator.evaluate_single_video(scores, "video_1", "summe")
    print(f"F-score: {fscore}")