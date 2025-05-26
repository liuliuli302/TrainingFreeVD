# TrainingFreeVD 使用指南

## 项目简介

TrainingFreeVD是一个无训练视频摘要生成系统，能够自动从视频中提取关键帧并生成摘要。

## 项目结构概述

```
TrainingFreeVD/
├── main.py                 # 简单版主程序
├── main_advanced.py        # 高级版主程序（支持命令行参数）
├── USAGE.md               # 本使用说明文件
├── README.md              # 项目说明
└── src/                   # 源代码目录
    ├── config/            # 配置文件
    │   └── config.py      # 各组件配置类
    ├── dataset/           # 数据集处理
    │   ├── dataset_builder.py
    │   └── vs_dataset.py
    ├── eval/              # 评估模块
    │   └── evaluator.py
    ├── llm/               # 大语言模型相关
    │   ├── captioner.py   # 视频描述生成
    │   ├── extractor.py   # 特征提取
    │   ├── llm_handler.py # LLM处理器
    │   └── llm_query.py   # LLM查询
    ├── pipeline/          # 主要流程
    │   └── pipeline.py    # 视频摘要生成流程
    └── util/              # 工具模块
        ├── constant.py    # 常量定义
        └── util.py        # 工具函数
```

## 主要功能

系统执行以下8个步骤来生成视频摘要：

1. **帧提取** - 从视频文件中提取关键帧
2. **数据集构建** - 构建处理数据集 
3. **视频字幕生成** - 使用LLaVA模型生成视频描述
4. **视觉特征提取** - 提取视频帧的视觉特征
5. **文本特征提取** - 提取视频描述的文本特征
6. **LLM查询** - 使用大语言模型对帧进行重要性评分
7. **语义相似度计算** - 计算帧之间的语义相似度
8. **最终评估** - 综合评分并生成最终摘要结果

## 快速开始

### 方法1: 使用简单版主程序

1. 编辑 `main.py` 文件，修改数据路径：
```python
config = VideoSummarizationPipelineConfig(
    base_data_dir="/your/data/path",     # 修改为你的数据目录
    base_output_dir="/your/output/path", # 修改为你的输出目录
    dataset_name="TVSum"                 # TVSum 或 SumMe
)
```

2. 运行程序：
```bash
python main.py
```

### 方法2: 使用高级版主程序（推荐）

直接通过命令行参数运行：

```bash
# 基本用法
python main_advanced.py --data_dir /path/to/data --output_dir /path/to/output

# 完整参数示例
python main_advanced.py \
    --data_dir /path/to/data \
    --output_dir /path/to/output \
    --dataset TVSum \
    --model_name lmms-lab/LLaVA-Video-7B-Qwen2 \
    --max_frames 64 \
    --frame_interval 15 \
    --clip_length 5 \
    --device auto
```

### 命令行参数说明

- `--data_dir`: 输入数据目录路径（必需）
- `--output_dir`: 输出结果目录路径（必需）
- `--dataset`: 数据集名称，可选 TVSum 或 SumMe（默认: TVSum）
- `--model_name`: LLM模型名称（默认: lmms-lab/LLaVA-Video-7B-Qwen2）
- `--max_frames`: 视频最大帧数（默认: 64）
- `--frame_interval`: 帧间隔（默认: 15）
- `--clip_length`: 视频片段长度（默认: 5）
- `--device`: 计算设备，可选 auto/cuda/cpu（默认: auto）

## 数据目录结构要求

你的数据目录应该具有以下结构：

```
data/
└── [数据集名称]/          # TVSum 或 SumMe
    ├── videos/           # 视频文件目录
    │   ├── video1.mp4
    │   ├── video2.mp4
    │   └── ...
    ├── annotations/      # 标注文件目录
    │   └── test.txt
    └── [数据集名称].h5    # HDF5格式的数据集文件
```

## 输出结果

程序执行完成后，会在输出目录生成以下结构：

```
output/
└── dataset/
    ├── result/
    │   ├── raw/              # 原始LLM输出
    │   ├── scores/           # 评分文件
    │   ├── similarity_scores/ # 相似度分数
    │   └── f1score/          # 最终F1分数
    └── [其他中间文件]
```

## 依赖要求

主要依赖包括：
- PyTorch
- Transformers
- OpenCV
- Decord
- BLIP-2
- LLaVA
- 其他深度学习相关包

## 注意事项

1. 首次运行会下载预训练模型，需要网络连接
2. 建议使用GPU加速，需要足够的显存（推荐8GB+）
3. 处理大型视频文件需要足够的磁盘空间
4. 确保数据目录结构符合要求

## 故障排除

如果遇到问题，请检查：
1. 数据目录结构是否正确
2. 必要的依赖是否已安装
3. GPU内存是否充足（如使用GPU）
4. 网络连接是否正常（下载模型时需要）
