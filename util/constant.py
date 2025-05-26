"""
静态变量常量文件
所有项目中的常量都定义在这里，采用全大写命名
"""
import numpy as np

# 数据集建造器相关常量
CLIP_TYPES = {"turn": "00", "jump": "01"}

# 视频文件扩展名
VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv')

# 默认分数模式
SCORE_PATTERN = r"Score:\s*(\d+\.\d+)"

# 默认提示词模板
DEFAULT_PROMPT_TEMPLATES = [
    "Summarize the main content and main events of the video in a concise and clear manner according to the order of events."
]

# 数据集建造提示词
DEFAULT_DATASET_PROMPT = (
    "You are a professional short film editor and director. "
    "Please score the frames divided based on theirs representativeness, "
    "diversity, and interest on a scale from 0 to 1. You may need to refer "
    "to the context for rating. And give the final score list like `[scores]`.\n "
    "without any extra text. You must output score."
)

# LLM查询默认提示词
DEFAULT_FIRST_PROMPT = (
    "You are a professional short film editor and director. "
    "Please score the frames divided based on theirs representativeness, "
    "diversity, and interest on a scale from 0 to 1. You may need to refer "
    "to the context for rating. And give the final score list like `[scores]`.\n "
    "without any extra text. You must output score."
)

DEFAULT_SECOND_PROMPT = (
    "Please provide a detailed analysis of each frame and give final scores."
)

# 数据集文件名
DATASET_FILE_NAMES = [
    "summe_dataset_jump.json",
    "summe_dataset_turn.json", 
    "tvsum_dataset_jump.json",
    "tvsum_dataset_turn.json"
]

# HDF5文件名
HDF5_FILE_NAMES = {
    "summe": "summe.h5",
    "tvsum": "tvsum.h5"
}

# 数据集评估方法
EVAL_METHODS = {
    "summe": "max",
    "tvsum": "avg"
}

# 相似度分数类型
SIMILARITY_SCORE_TYPES = [
    "max_p_max_m", 
    "max_p_mean_m", 
    "mean_p_max_m", 
    "mean_p_mean_m"
]

# LLM模型类型
LLM_TYPES = ["jump", "turn"]

# Alpha值范围（用于组合分数）
ALPHA_RANGE = [round(x, 1) for x in np.arange(0.1, 1.1, 0.1)]

# 默认帧间隔
DEFAULT_FRAME_INTERVAL = 15

# 默认最大帧数
DEFAULT_MAX_FRAMES = 64

# 默认对话模板
DEFAULT_CONV_TEMPLATE = "qwen_1_5"

# 默认步长
DEFAULT_STRIDE = 15

# 默认批次大小
DEFAULT_BATCH_SIZE = 128

# 图片文件扩展名
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

# 默认摘要选择比例
DEFAULT_SUMMARY_RATIO = 0.15

# 数值计算常量
EPSILON = 1e-8

# 路径后缀
PATH_SUFFIXES = {
    "frames": "frames",
    "videos": "videos", 
    "captions": "captions",
    "features": "features",
    "visual": "visual",
    "text": "text",
    "annotations": "annotations"
}

# 数据集名称映射
DATASET_MAPPING = {
    "summe": "SumMe",
    "tvsum": "TVSum"
}
