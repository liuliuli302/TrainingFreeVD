import argparse
from pathlib import Path
import json
import numpy as np
import copy
import os
import re
from tqdm import tqdm
from typing import Dict, List, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from config.config import LLMQueryConfig
from .llm_handler import LLMHandler
import h5py


class LLMQuery:
    """Query LLM for video summarization scores"""
    
    def __init__(self, config: 'LLMQueryConfig', llm_handler: LLMHandler):
        self.config = config
        self.llm_handler = llm_handler
        # 使用配置中的常量
        self.score_pattern = config.score_pattern
        
        # Query prompts from config
        self.first_prompt = config.first_prompt
        self.second_prompt = config.second_prompt
    
    def load_datasets(self, dataset_dir: Path) -> List[Path]:
        """Load dataset JSON files"""
        # 使用配置中的数据集文件名
        json_files = [
            dataset_dir / "SumMe" / self.config.dataset_file_names[0],  # summe_dataset_jump.json
            dataset_dir / "SumMe" / self.config.dataset_file_names[1],  # summe_dataset_turn.json
            dataset_dir / "TVSum" / self.config.dataset_file_names[2],  # tvsum_dataset_jump.json
            dataset_dir / "TVSum" / self.config.dataset_file_names[3],  # tvsum_dataset_turn.json
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
        summe_hdf = dataset_dir / "SumMe" / self.config.hdf5_file_names["summe"]
        tvsum_hdf = dataset_dir / "TVSum" / self.config.hdf5_file_names["tvsum"]
        
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
            index = int(int(os.path.basename(path).split(".")[0]) / frame_interval)
            score_index.append(index)
        return score_index
    
    def extract_scores_from_response(self, llm_response: str) -> List[float]:
        """Extract scores from LLM response"""
        scores = re.findall(self.score_pattern, llm_response)
        scores = [float(score) for score in scores]
        return scores
    
    def query_single_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Query LLM for a single sample"""
        try:
            # Preprocess images
            images = self.llm_handler.preprocess_images(sample["images"])
            video_time = sample["video_time"]
            frame_time = sample["frame_time"]
            
            # Multi-turn conversation
            prompts = [self.first_prompt, self.second_prompt]
            responses = self.llm_handler.multi_turn_conversation(
                images, prompts, video_time, frame_time, max_new_tokens=1000
            )
            
            # Prepare result
            result_sample = copy.deepcopy(sample)
            result_sample["llm_out"] = responses[-1]  # Use the last response
            
            return result_sample
            
        except Exception as e:
            print(f"Error processing sample {sample.get('id', 'unknown')}: {str(e)}")
            result_sample = copy.deepcopy(sample)
            result_sample["llm_out"] = f"Error: {str(e)}"
            return result_sample
    
    def process_dataset_file(self, dataset_file: Path, result_dir: Path) -> Path:
        """Process a single dataset file"""
        result_file_name = str(dataset_file.name).split('.')[0] + "_result.json"
        result_file = result_dir / result_file_name
        
        # Load dataset
        with open(dataset_file, "r") as f:
            dataset = json.load(f)
        
        results = []
        print(f"Processing {dataset_file.name}...")
        
        for sample in tqdm(dataset, desc="Samples"):
            result_sample = self.query_single_sample(sample)
            results.append(result_sample)
        
        # Save results
        with open(result_file, "w") as f:
            json.dump(results, f, indent=4)
        
        print(f"Results saved to {result_file}")
        return result_file
    
    def process_results_to_scores(self, result_files: List[Path], 
                                 score_list: Dict, frame_interval: int = 15) -> Dict:
        """Process LLM results to extract scores"""
        result_dict = {}
        
        for file in result_files:
            with open(file, "r") as f:
                result = json.load(f)
            
            file_name = file.stem
            dataset_name = file_name.split("_")[0]
            
            if dataset_name not in score_list:
                print(f"Warning: Dataset {dataset_name} not found in score_list")
                continue
            
            check_out = []
            scores_list_tmp = copy.deepcopy(score_list[dataset_name])
            
            for sample in result:
                try:
                    # Extract video name
                    video_name = "_".join(sample["id"].split("_")[1:3])
                    
                    if video_name not in scores_list_tmp:
                        print(f"Warning: Video {video_name} not found in score list")
                        continue
                    
                    # Get indices and scores
                    images = sample["images"]
                    index = self.get_score_index_from_image_list(images, frame_interval)
                    index = np.array(index)
                    
                    llm_out = sample["llm_out"]
                    scores = self.extract_scores_from_response(llm_out)
                    scores = np.array(scores)
                    
                    # Handle single score case
                    if isinstance(scores, float):
                        scores = [scores]
                    
                    # Validate scores length
                    if len(scores) == len(index):
                        check_out.append(1)
                    else:
                        if len(scores) < len(index):
                            # Pad with zeros
                            scores = np.concatenate(
                                (scores, np.zeros(len(index) - len(scores)))
                            )
                        elif len(scores) > len(index):
                            # Truncate
                            scores = scores[:len(index)]
                        check_out.append(0)
                    
                    # Update scores
                    scores_list_tmp[video_name][index] = scores
                    
                except Exception as e:
                    print(f"Error processing sample {sample.get('id', 'unknown')}: {str(e)}")
                    check_out.append(0)
            
            # Convert to list for JSON serialization
            for video_name in scores_list_tmp:
                scores_list_tmp[video_name] = scores_list_tmp[video_name].tolist()
            
            # Print validation statistics
            check_out = np.array(check_out)
            right = check_out.sum()
            length = len(check_out)
            print(f"{file_name} - Score extraction rate: {right}/{length}")
            
            result_dict[file_name] = scores_list_tmp
        
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
        result_dict = self.process_results_to_scores(result_files, score_list, frame_interval)
        
        # Save final scores
        scores_json = scores_dir / "raw_llm_out_scores.json"
        with open(scores_json, "w") as f:
            json.dump(result_dict, f, separators=(",", ": "), indent=4)
        
        print(f"Final scores saved to {scores_json}")


def main():
    parser = argparse.ArgumentParser(description="Query LLM for video summarization")
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