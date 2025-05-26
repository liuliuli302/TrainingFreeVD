# TrainingFreeVD
Training Free Video Summarization - 无训练视频摘要生成系统

## 功能特性

- 支持命令行参数运行，便于实验管理
- 自动创建独立的实验目录
- 完整的视频摘要生成流程
- 实验结果自动保存和管理
- 支持多种数据集（TVSum、SumMe）

## 快速开始

### 基本用法

1. **指定实验名称运行：**
```bash
python main.py --exam my_experiment1
```

2. **自动生成实验名称：**
```bash
python main.py
# 将自动生成 exam1, exam2, exam3... 等实验名称
```

### 实验管理

所有实验结果都保存在 `out/` 目录下：

```
out/
├── my_experiment1/           # 实验目录
│   ├── my_experiment1_experiment_info.json    # 实验配置信息
│   ├── my_experiment1_final_results.json      # 最终结果
│   ├── my_experiment1_pipeline.log           # 运行日志
│   ├── dataset/                              # 数据集处理结果
│   └── ...                                   # 其他中间结果
├── exam1/                    # 自动生成的实验目录
└── exam2/
```

### 实验管理工具

使用内置的实验管理工具查看和管理实验：

```bash
# 列出所有实验
python src/util/experiment_manager.py list

# 查看特定实验详情
python src/util/experiment_manager.py show my_experiment1

# 删除实验
python src/util/experiment_manager.py delete my_experiment1
```

## 安全特性

- **重复实验名检测：** 如果指定的实验名已存在，程序会报错退出，避免覆盖现有实验
- **自动递增命名：** 未指定实验名时，自动查找下一个可用的编号
- **完整日志记录：** 每个实验都有独立的日志文件

## 配置说明

在 `main.py` 中修改以下配置：

```python
config = VideoSummarizationPipelineConfig(
    base_data_dir="/path/to/your/data",      # 修改为你的数据目录
    base_output_dir=exam_dir,                # 自动设置为实验目录
    dataset_name="TVSum"                     # 可选: TVSum 或 SumMe
)
```

## 流程步骤

1. **帧提取** - 从视频中提取关键帧
2. **数据集构建** - 构建训练数据集
3. **视频字幕生成** - 使用LLaVA模型生成视频描述
4. **视觉特征提取** - 提取视觉特征向量
5. **文本特征提取** - 提取文本特征向量
6. **LLM查询** - 使用大语言模型进行推理
7. **语义相似度计算** - 计算多模态相似度
8. **最终评估** - 生成最终的摘要评分

## 示例输出

运行成功后，你会看到类似的输出：

```
============================================================
TrainingFreeVD - 无训练视频摘要生成系统
============================================================
已创建实验目录: /root/TrainingFreeVD/out/my_experiment1
实验名称: my_experiment1
实验目录: /root/TrainingFreeVD/out/my_experiment1
数据目录: /path/to/your/data
输出目录: /root/TrainingFreeVD/out/my_experiment1
数据集: TVSum
------------------------------------------------------------
开始运行视频摘要生成流程...
...
============================================================
流程执行完成！
实验名称: my_experiment1
实验结果保存在: /root/TrainingFreeVD/out/my_experiment1
最终结果: {...}
============================================================
```
