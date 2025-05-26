#!/usr/bin/env python3
"""
实验管理工具
用于管理和查看实验结果
"""

import os
import json
import glob
from datetime import datetime
from pathlib import Path


class ExperimentManager:
    """实验管理器"""
    
    def __init__(self, out_dir="/root/TrainingFreeVD/out"):
        self.out_dir = out_dir
    
    def list_experiments(self):
        """列出所有实验"""
        if not os.path.exists(self.out_dir):
            print("No experiments found (out directory doesn't exist)")
            return []
        
        experiment_dirs = [d for d in os.listdir(self.out_dir) 
                          if os.path.isdir(os.path.join(self.out_dir, d))]
        
        if not experiment_dirs:
            print("No experiments found")
            return []
        
        experiments = []
        print(f"\n实验列表 (共 {len(experiment_dirs)} 个实验):")
        print("=" * 60)
        
        for exp_dir in sorted(experiment_dirs):
            exp_path = os.path.join(self.out_dir, exp_dir)
            info_file = os.path.join(exp_path, f"{exp_dir}_experiment_info.json")
            results_file = os.path.join(exp_path, f"{exp_dir}_final_results.json")
            
            exp_info = {
                "name": exp_dir,
                "path": exp_path,
                "created": "未知",
                "status": "运行中",
                "results": None
            }
            
            # 读取实验信息
            if os.path.exists(info_file):
                try:
                    with open(info_file, 'r', encoding='utf-8') as f:
                        info = json.load(f)
                        exp_info["created"] = info.get("created_at", "未知")
                except:
                    pass
            
            # 读取结果信息
            if os.path.exists(results_file):
                try:
                    with open(results_file, 'r', encoding='utf-8') as f:
                        results = json.load(f)
                        exp_info["status"] = results.get("status", "已完成")
                        exp_info["results"] = results.get("results")
                except:
                    pass
            
            experiments.append(exp_info)
            
            # 显示实验信息
            status_icon = "✓" if exp_info["status"] == "completed" else "○"
            print(f"{status_icon} {exp_dir}")
            print(f"   创建时间: {exp_info['created'][:19] if exp_info['created'] != '未知' else '未知'}")
            print(f"   状态: {exp_info['status']}")
            print(f"   路径: {exp_path}")
            print()
        
        return experiments
    
    def get_experiment_details(self, experiment_name):
        """获取特定实验的详细信息"""
        exp_path = os.path.join(self.out_dir, experiment_name)
        
        if not os.path.exists(exp_path):
            print(f"实验 '{experiment_name}' 不存在")
            return None
        
        print(f"\n实验详情: {experiment_name}")
        print("=" * 60)
        
        # 实验信息
        info_file = os.path.join(exp_path, f"{experiment_name}_experiment_info.json")
        if os.path.exists(info_file):
            try:
                with open(info_file, 'r', encoding='utf-8') as f:
                    info = json.load(f)
                    print("实验配置:")
                    for key, value in info.get("configuration", {}).items():
                        print(f"  {key}: {value}")
                    print(f"创建时间: {info.get('created_at', '未知')}")
            except Exception as e:
                print(f"读取实验信息失败: {e}")
        
        print()
        
        # 结果信息
        results_file = os.path.join(exp_path, f"{experiment_name}_final_results.json")
        if os.path.exists(results_file):
            try:
                with open(results_file, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                    print("实验结果:")
                    print(f"  状态: {results.get('status', '未知')}")
                    print(f"  完成时间: {results.get('completed_at', '未知')}")
                    if results.get('results'):
                        print(f"  结果: {results['results']}")
            except Exception as e:
                print(f"读取实验结果失败: {e}")
        else:
            print("实验尚未完成或结果文件不存在")
        
        print()
        
        # 文件列表
        print("实验文件:")
        for root, dirs, files in os.walk(exp_path):
            level = root.replace(exp_path, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files:
                file_path = os.path.join(root, file)
                file_size = os.path.getsize(file_path)
                size_str = self._format_size(file_size)
                print(f"{subindent}{file} ({size_str})")
    
    def _format_size(self, size_bytes):
        """格式化文件大小"""
        if size_bytes == 0:
            return "0B"
        
        size_names = ["B", "KB", "MB", "GB", "TB"]
        i = 0
        while size_bytes >= 1024 and i < len(size_names) - 1:
            size_bytes /= 1024.0
            i += 1
        
        return f"{size_bytes:.1f}{size_names[i]}"
    
    def delete_experiment(self, experiment_name, confirm=True):
        """删除实验"""
        exp_path = os.path.join(self.out_dir, experiment_name)
        
        if not os.path.exists(exp_path):
            print(f"实验 '{experiment_name}' 不存在")
            return False
        
        if confirm:
            response = input(f"确定要删除实验 '{experiment_name}' 吗? (y/N): ")
            if response.lower() != 'y':
                print("取消删除")
                return False
        
        try:
            import shutil
            shutil.rmtree(exp_path)
            print(f"已删除实验: {experiment_name}")
            return True
        except Exception as e:
            print(f"删除实验失败: {e}")
            return False


def main():
    """命令行工具主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="实验管理工具")
    parser.add_argument("command", choices=["list", "show", "delete"], 
                       help="命令: list(列出实验), show(显示详情), delete(删除实验)")
    parser.add_argument("experiment", nargs="?", help="实验名称")
    parser.add_argument("--force", action="store_true", help="强制删除，不确认")
    
    args = parser.parse_args()
    
    manager = ExperimentManager()
    
    if args.command == "list":
        manager.list_experiments()
    elif args.command == "show":
        if not args.experiment:
            print("请指定实验名称")
            return
        manager.get_experiment_details(args.experiment)
    elif args.command == "delete":
        if not args.experiment:
            print("请指定要删除的实验名称")
            return
        manager.delete_experiment(args.experiment, confirm=not args.force)


if __name__ == "__main__":
    main()
