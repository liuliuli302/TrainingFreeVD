#!/usr/bin/env python3
"""
TrainingFreeVD Main Entry Point
视频摘要生成主程序入口
"""

import sys
import os
import argparse
import glob
import re

# 添加src目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.append(src_dir)

from src.config.config import VideoSummarizationPipelineConfig
from src.pipeline.pipeline import VideoSummarizationPipeline


def parse_arguments():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(
        description='TrainingFreeVD - 无训练视频摘要生成系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python main.py --exam my_experiment1
  python main.py  # 使用自动递增的实验名称 (exam1, exam2, ...)
        """
    )
    
    parser.add_argument(
        '--exam', 
        type=str, 
        default=None,
        help='实验名称。如果不指定，将自动生成 exam1, exam2, ... 等名称'
    )
    
    return parser.parse_args()


def get_next_exam_name(out_dir):
    """
    获取下一个可用的实验名称 (exam1, exam2, ...)
    """
    existing_dirs = glob.glob(os.path.join(out_dir, "exam*"))
    existing_numbers = []
    
    for dir_path in existing_dirs:
        dir_name = os.path.basename(dir_path)
        match = re.match(r'exam(\d+)$', dir_name)
        if match:
            existing_numbers.append(int(match.group(1)))
    
    if not existing_numbers:
        return "exam1"
    
    next_number = max(existing_numbers) + 1
    return f"exam{next_number}"


def setup_experiment_directory(exam_name):
    """
    设置实验目录，检查重复并创建目录
    """
    out_dir = os.path.join(current_dir, "out")
    
    # 确保 out 目录存在
    os.makedirs(out_dir, exist_ok=True)
    
    # 如果没有指定实验名称，自动生成
    if exam_name is None:
        exam_name = get_next_exam_name(out_dir)
        print(f"未指定实验名称，自动生成: {exam_name}")
    
    # 检查实验目录是否已存在
    exam_dir = os.path.join(out_dir, exam_name)
    if os.path.exists(exam_dir):
        print(f"错误: 实验目录 '{exam_name}' 已存在于 {exam_dir}")
        print("请使用不同的实验名称或删除现有目录。")
        sys.exit(1)
    
    # 创建实验目录
    os.makedirs(exam_dir, exist_ok=True)
    print(f"已创建实验目录: {exam_dir}")
    
    return exam_dir, exam_name


def main():
    """
    主函数：解析命令行参数，创建实验目录，配置并运行视频摘要生成pipeline
    """
    # 解析命令行参数
    args = parse_arguments()
    
    print("=" * 60)
    print("TrainingFreeVD - 无训练视频摘要生成系统")
    print("=" * 60)
    
    # 设置实验目录
    exam_dir, exam_name = setup_experiment_directory(args.exam)
    
    # 创建配置，使用实验目录作为输出目录
    # 用户可以根据需要修改这些路径
    config = VideoSummarizationPipelineConfig(
        base_data_dir="/path/to/your/data",  # 修改为你的数据目录
        base_output_dir=exam_dir,  # 使用实验目录作为输出目录
        dataset_name="TVSum",  # 可选: TVSum 或 SumMe
        exam_name=exam_name  # 添加实验名称到配置
    )
    
    print(f"实验名称: {exam_name}")
    print(f"实验目录: {exam_dir}")
    print(f"数据目录: {config.base_data_dir}")
    print(f"输出目录: {config.base_output_dir}")
    print(f"数据集: {config.dataset_name}")
    print("-" * 60)
    
    try:
        # 创建pipeline实例
        pipeline = VideoSummarizationPipeline(config)
        
        # 运行完整的视频摘要生成流程
        print("开始运行视频摘要生成流程...")
        results = pipeline.run()
        
        print("=" * 60)
        print("流程执行完成！")
        print(f"实验名称: {exam_name}")
        print(f"实验结果保存在: {exam_dir}")
        print(f"最终结果: {results}")
        print("=" * 60)
        
    except Exception as e:
        print(f"执行过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    except Exception as e:
        print(f"执行过程中出现错误: {str(e)}")
        print("请检查配置和依赖是否正确安装")
        sys.exit(1)


if __name__ == "__main__":
    main()
