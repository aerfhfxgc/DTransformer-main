#!/usr/bin/env python
"""
生成 datasets_folds.toml 配置文件

为每个数据集的所有 fold 和 final 切分生成配置条目
"""

import os
import argparse
import tomlkit
from pathlib import Path


def generate_folds_config(
    dataset_name: str,
    base_config: dict,
    n_folds: int,
    data_dir: str = 'data'
) -> dict:
    """
    为指定数据集生成 fold 配置
    
    Args:
        dataset_name: 数据集名称
        base_config: 原始数据集配置
        n_folds: fold 数量
        data_dir: 数据目录
    
    Returns:
        包含所有 fold 配置的字典
    """
    folds_config = {}
    
    # 生成每个 fold 的配置
    for i in range(1, n_folds + 1):
        fold_name = f"{dataset_name}_fold{i}"
        fold_config = {
            'train': f"{dataset_name}/folds/fold{i}/train.txt",
            'valid': f"{dataset_name}/folds/fold{i}/valid.txt",
            'test': base_config['test'],
            'n_questions': base_config['n_questions'],
            'n_pid': base_config['n_pid'],
            'inputs': base_config['inputs']
        }
        
        # 如果有 seq_len，也复制过来
        if 'seq_len' in base_config:
            fold_config['seq_len'] = base_config['seq_len']
        
        folds_config[fold_name] = fold_config
    
    # 生成 final 配置
    final_name = f"{dataset_name}_final"
    final_config = {
        'train': f"{dataset_name}/final/train.txt",
        'valid': f"{dataset_name}/final/valid.txt",
        'test': base_config['test'],
        'n_questions': base_config['n_questions'],
        'n_pid': base_config['n_pid'],
        'inputs': base_config['inputs']
    }
    
    if 'seq_len' in base_config:
        final_config['seq_len'] = base_config['seq_len']
    
    folds_config[final_name] = final_config
    
    return folds_config


def main():
    parser = argparse.ArgumentParser(
        description='生成 datasets_folds.toml 配置文件'
    )
    parser.add_argument(
        '--datasets',
        nargs='+',
        help='要生成配置的数据集列表（如果不指定，则为所有数据集）'
    )
    parser.add_argument(
        '--n_folds',
        type=int,
        default=5,
        help='fold 数量（默认：5）'
    )
    parser.add_argument(
        '--input_config',
        default='data/datasets.toml',
        help='输入配置文件路径（默认：data/datasets.toml）'
    )
    parser.add_argument(
        '--output_config',
        default='data/datasets_folds.toml',
        help='输出配置文件路径（默认：data/datasets_folds.toml）'
    )
    parser.add_argument(
        '--data_dir',
        default='data',
        help='数据目录（默认：data）'
    )
    
    args = parser.parse_args()
    
    # 读取原始配置
    print(f"读取原始配置: {args.input_config}")
    with open(args.input_config, 'r', encoding='utf-8') as f:
        original_config = tomlkit.load(f)
    
    # 确定要处理的数据集
    if args.datasets:
        datasets_to_process = args.datasets
    else:
        datasets_to_process = list(original_config.keys())
    
    print(f"将为以下数据集生成配置: {', '.join(datasets_to_process)}")
    
    # 生成新配置
    new_config = tomlkit.document()
    
    for dataset_name in datasets_to_process:
        if dataset_name not in original_config:
            print(f"警告: 数据集 '{dataset_name}' 未在原始配置中找到，跳过")
            continue
        
        print(f"\n生成 {dataset_name} 的配置...")
        base_config = original_config[dataset_name]
        
        folds_config = generate_folds_config(
            dataset_name=dataset_name,
            base_config=base_config,
            n_folds=args.n_folds,
            data_dir=args.data_dir
        )
        
        # 添加到新配置
        for name, config in folds_config.items():
            new_config[name] = config
            print(f"  - {name}")
    
    # 写入输出文件
    print(f"\n写入配置文件: {args.output_config}")
    os.makedirs(os.path.dirname(args.output_config) or '.', exist_ok=True)
    
    with open(args.output_config, 'w', encoding='utf-8') as f:
        tomlkit.dump(new_config, f)
    
    print(f"✓ 配置文件生成成功！")
    print(f"  总共生成 {len(new_config)} 个配置条目")


if __name__ == '__main__':
    main()
