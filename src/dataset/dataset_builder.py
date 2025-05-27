import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Union, TYPE_CHECKING
import h5py
import numpy as np
from decord import VideoReader, cpu
from torch.utils.data import Dataset, DataLoader

from src.config.config import DatasetBuilderConfig


class DatasetBuilder:
    """优雅的数据集构建器，支持SumMe和TVSum数据集的处理"""
    
    def __init__(self, config: DatasetBuilderConfig):
        self.config = config
        self.data_dir = Path(config.data_dir)
        self.save_dir = Path(config.save_dir)
        self.clip_length = config.clip_length
        # 使用配置中的常量
        self.clip_types = config.clip_types
        self.prompt = config.dataset_prompt
        
    def load_hdf5_as_dict(self, file_path: Path) -> Dict:
        """将HDF5文件递归转换为字典"""
        with h5py.File(file_path, "r") as h5_file:
            def _convert_recursively(h5_obj):
                if isinstance(h5_obj, h5py.Group):
                    return {key: _convert_recursively(h5_obj[key]) for key in h5_obj.keys()}
                elif isinstance(h5_obj, h5py.Dataset):
                    return h5_obj[()]
                else:
                    raise TypeError(f"Unsupported h5py object type: {type(h5_obj)}")
            return _convert_recursively(h5_file)
    
    def create_clips_turn(self, picks: List[int]) -> Tuple[List[List[int]], int]:
        """顺序划分clips，每clip_length帧为一组"""
        remainder = len(picks) % self.clip_length
        valid_length = len(picks) - remainder
        clips = [picks[i:i+self.clip_length] for i in range(0, valid_length, self.clip_length)]
        return clips, remainder
    
    def create_clips_jump(self, picks: List[int]) -> Tuple[List[List[int]], int]:
        """跳跃划分clips，将picks分为clip_length段后跳帧采样"""
        num_samples = len(picks) // self.clip_length
        remainder = len(picks) % num_samples
        clips = []
        for i in range(num_samples):
            indices = [picks[i + j * num_samples] for j in range(self.clip_length)]
            clips.append(indices)
        return clips, remainder
    
    def generate_sample_id(self, dataset_name: str, video_name: str, 
                          clip_type: str, sample_id: int, remainder: int) -> str:
        """生成样本ID"""
        clip_code = self.clip_types[clip_type]
        return f"{dataset_name}_{video_name}_{clip_code}{sample_id:04d}{remainder:02d}"
    
    def get_video_info(self, video_path: Path) -> Tuple[float, float]:
        """获取视频信息：总时长和帧率"""
        vr = VideoReader(str(video_path), ctx=cpu(0), num_threads=1)
        fps = round(vr.get_avg_fps())
        video_time = len(vr) / fps
        return video_time, fps
    
    def format_frame_times(self, frames: List[int], fps: float) -> str:
        """格式化帧时间信息"""
        frame_times = [f"{frame/fps:.2f}s" for frame in frames]
        return ",".join(frame_times)
    
    def create_sample(self, sample_id: str, image_paths: List[str], 
                     video_time: float, frame_time: str) -> Dict:
        """创建单个样本"""
        return {
            "id": sample_id,
            "images": image_paths,
            "prompt": self.prompt,
            "video_time": video_time,
            "frame_time": frame_time
        }
    
    def process_dataset(self, dataset_name: str) -> Dict[str, List[Dict]]:
        """处理单个数据集（SumMe或TVSum）"""
        # 构建路径
        dataset_dir = self.data_dir / dataset_name
        h5_path = dataset_dir / f"{dataset_name.lower()}.h5"
        json_path = dataset_dir / "video_name_dict.json"
        frames_dir = dataset_dir / "frames"
        videos_dir = dataset_dir / "videos"
        
        # 加载数据
        dataset_dict = self.load_hdf5_as_dict(h5_path)
        with open(json_path, "r") as f:
            name_dict = json.load(f)
        name_dict_reverse = {v: k for k, v in name_dict.items()}
        
        samples = {"turn": [], "jump": []}
        
        for video_name in dataset_dict.keys():
            video_name_real = name_dict_reverse[video_name]
            video_path = videos_dir / f"{video_name_real}.mp4"
            frame_dir = frames_dir / video_name_real
            
            # 获取视频信息
            video_time, fps = self.get_video_info(video_path)
            picks = dataset_dict[video_name]["picks"]
            
            # 处理两种clip类型
            for clip_type in ["turn", "jump"]:
                if clip_type == "turn":
                    clips, remainder = self.create_clips_turn(picks)
                else:
                    clips, remainder = self.create_clips_jump(picks)
                
                for i, clip in enumerate(clips):
                    sample_id = self.generate_sample_id(dataset_name, video_name, clip_type, i, remainder)
                    image_paths = [str(frame_dir / f"{frame:06d}.jpg") for frame in clip]
                    frame_time = self.format_frame_times(clip, fps)
                    
                    sample = self.create_sample(sample_id, image_paths, video_time, frame_time)
                    samples[clip_type].append(sample)
        
        return samples
    
    def save_datasets(self, dataset_samples: Dict[str, Dict[str, List[Dict]]]):
        """保存数据集到JSON文件"""
        for dataset_name, samples in dataset_samples.items():
            output_dir = self.save_dir / dataset_name
            output_dir.mkdir(parents=True, exist_ok=True)
            
            for clip_type, data in samples.items():
                output_file = output_dir / f"{dataset_name.lower()}_dataset_{clip_type}.json"
                with open(output_file, "w") as f:
                    json.dump(data, f, indent=4)
                print(f"Saved {len(data)} samples to {output_file}")
    
    def build(self):
        """构建完整数据集"""
        print("Starting dataset building...")
        
        dataset_samples = {}
        for dataset_name in ["SumMe", "TVSum"]:
            print(f"Processing {dataset_name}...")
            dataset_samples[dataset_name] = self.process_dataset(dataset_name)
            
        print("Saving datasets...")
        self.save_datasets(dataset_samples)
        print("Dataset building completed!")


class VideoSummarizationDataset(Dataset):
    def __init__(
        self,
        root_path="./data",
        dataset_name="SumMe",
    ):
        super().__init__()
        self.root_path = root_path
        self.dataset_name = dataset_name
        self.dataset_dir = Path(root_path, dataset_name)
        self.frames_dir = Path(self.dataset_dir, "frames")

        self.data_list, self.video_name_dict = self._load_data()

        # Invert the values and keys in the self.video_name_dict
        self.video_name_dict_inv = {v: k for k,
                                    v in self.video_name_dict.items()}
        # pprint(self.video_name_dict_inv)

    def __len__(self):
        return len(self.video_name_dict)

    def __getitem__(self, idx):
        video_name_idx = f"video_{idx + 1}"
        video_name_real = self.video_name_dict_inv[video_name_idx]
        video_frames_dir = Path(self.frames_dir, video_name_real)

        video_info = self.data_list[video_name_idx]

        picks = video_info["picks"]
        keys = list(video_info.keys())
        # Convert picks to 6-digit integer.
        picks = [f"{pick:06d}" for pick in picks]
        # Gets all file names from picks.
        frame_file_paths = [
            str(Path(video_frames_dir, f"{pick}.jpg")) for pick in picks]

        video_info["frame_file_paths"] = frame_file_paths
        video_info["video_name"] = video_name_real

        # Debug info.
        # pprint(frame_file_paths)
        # pprint(keys)

        return video_info

    def _load_data(self):
        """
        Load data from `self.data_path`.
        """
        # 1 Load hdf file to dict.
        dataset_name_lower = self.dataset_name.lower()
        hdf_file_path = Path(self.dataset_dir, f"{dataset_name_lower}.h5")
        hdf_file = h5py.File(hdf_file_path, "r")

        hdf_dict = hdf5_to_dict(hdf_file)
        video_names = list(hdf_dict.keys())
        keys = list(hdf_dict["video_1"].keys())

        # 2 Load video_name dict.
        video_name_dict_file_path = Path(
            self.dataset_dir, "video_name_dict.json")
        with open(video_name_dict_file_path, "r") as f:
            video_name_dict = json.load(f)

        return hdf_dict, video_name_dict


def apply_conversation_template(llm_name: str, num_images: int, prompt: str) -> List[Dict]:
    """生成对话模板（保持原有功能）"""
    if llm_name == "llava-next":
        content = [{"type": "text", "text": prompt}]
        for i in range(num_images):
            content.extend([
                {"type": "text", "text": f"Frame {i}"},
                {"type": "image"}
            ])
        return [{"role": "user", "content": content}]
    return []


def hdf5_to_dict(hdf5_file):
    def recursively_convert(h5_obj):
        if isinstance(h5_obj, h5py.Group):
            return {key: recursively_convert(h5_obj[key]) for key in h5_obj.keys()}
        elif isinstance(h5_obj, h5py.Dataset):
            return h5_obj[()]
        else:
            raise TypeError("Unsupported h5py object type")

    return recursively_convert(hdf5_file)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Build dataset for video summarization")
    parser.add_argument("--dataset_dir", type=str, required=True, 
                       help="Directory path to the dataset")
    parser.add_argument("--dataset_save_dir", type=str, required=True,
                       help="Directory to save the processed dataset")
    parser.add_argument("--clip_length", type=int, default=5,
                       help="Length of each clip")
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    builder = DatasetBuilder(
        data_dir=args.dataset_dir,
        save_dir=args.dataset_save_dir,
        clip_length=args.clip_length
    )
    
    builder.build()


if __name__ == "__main__":
    main()