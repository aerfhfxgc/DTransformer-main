#!/usr/bin/env python
"""
为知识追踪数据集生成 K-fold 交叉验证数据切分

此脚本确保：
1. 按样本块（group = len(inputs) + 1）为单位切分数据
2. 不会破坏学生序列的完整性
3. 生成的每个fold都包含独立的train和valid集
"""

import os
import argparse
from pathlib import Path
import random
import tomlkit


def read_groups_from_file(filepath, group_size):
    """
    按 group 读取数据文件
    
    Args:
        filepath: 数据文件路径
        group_size: 每个样本块的行数 (len(inputs) + 1)
    
    Returns:
        groups: 样本块列表，每个元素是包含 group_size 行的列表
    """
    groups = []
    current_group = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            current_group.append(line)
            if len(current_group) == group_size:
                groups.append(current_group)
                current_group = []
    
    if current_group:
        raise ValueError(
            f"文件 {filepath} 的行数不能被 group_size={group_size} 整除。"
            f"剩余 {len(current_group)} 行未处理。"
            f"请检查数据格式或 inputs 配置是否正确。"
        )
    
    return groups


def write_groups_to_file(groups, filepath):
    """
    将样本块写入文件
    
    Args:
        groups: 样本块列表
        filepath: 输出文件路径
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        for group in groups:
            for line in group:
                f.write(line)


def create_k_folds(groups, k, seed=42):
    """
    创建 K-fold 切分
    
    Args:
        groups: 样本块列表
        k: fold 数量
        seed: 随机种子
    
    Returns:
        folds: 列表，每个元素是 (train_groups, valid_groups) 元组
    """
    # 固定随机种子确保可复现
    random.seed(seed)
    shuffled_groups = groups.copy()
    random.shuffle(shuffled_groups)
    
    n_samples = len(shuffled_groups)
    fold_size = n_samples // k
    
    folds = []
    for i in range(k):
        # 计算当前fold的验证集范围
        valid_start = i * fold_size
        if i == k - 1:  # 最后一个fold包含所有剩余样本
            valid_end = n_samples
        else:
            valid_end = (i + 1) * fold_size
        
        # 划分验证集和训练集
        valid_groups = shuffled_groups[valid_start:valid_end]
        train_groups = shuffled_groups[:valid_start] + shuffled_groups[valid_end:]
        
        folds.append((train_groups, valid_groups))
    
    return folds


def main():
    parser = argparse.ArgumentParser(
        description='为知识追踪数据集生成 K-fold 交叉验证数据'
    )
    parser.add_argument(
        '-d', '--dataset',
        required=True,
        help='数据集名称（如 assist09）'
    )
    parser.add_argument(
        '-k', '--n_folds',
        type=int,
        default=5,
        help='fold 数量（默认：5）'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子（默认：42）'
    )
    parser.add_argument(
        '--data_dir',
        default='data',
        help='数据目录（默认：data）'
    )
    parser.add_argument(
        '--config',
        default='data/datasets.toml',
        help='数据集配置文件路径（默认：data/datasets.toml）'
    )
    
    args = parser.parse_args()
    
    # 读取数据集配置
    print(f"读取配置文件: {args.config}")
    with open(args.config, 'r', encoding='utf-8') as f:
        datasets = tomlkit.load(f)
    
    if args.dataset not in datasets:
        raise ValueError(
            f"数据集 '{args.dataset}' 未在配置文件中找到。"
            f"可用数据集: {list(datasets.keys())}"
        )
    
    dataset_config = datasets[args.dataset]
    
    # 计算 group size
    inputs = dataset_config['inputs']
    group_size = len(inputs) + 1
    
    print(f"\n数据集: {args.dataset}")
    print(f"inputs: {inputs}")
    print(f"group_size: {group_size}")
    
    # 读取原始训练数据
    train_file = os.path.join(args.data_dir, dataset_config['train'])
    print(f"\n读取训练数据: {train_file}")
    
    groups = read_groups_from_file(train_file, group_size)
    n_samples = len(groups)
    n_lines = n_samples * group_size
    
    print(f"总样本块数: {n_samples}")
    print(f"总行数: {n_lines}")
    
    # 生成 K-fold 切分
    print(f"\n生成 {args.n_folds}-fold 切分 (seed={args.seed})...")
    folds = create_k_folds(groups, args.n_folds, args.seed)
    
    # 输出每个fold
    output_base = os.path.join(args.data_dir, args.dataset, 'folds')
    
    for i, (train_groups, valid_groups) in enumerate(folds, 1):
        fold_dir = os.path.join(output_base, f'fold{i}')
        
        train_file = os.path.join(fold_dir, 'train.txt')
        valid_file = os.path.join(fold_dir, 'valid.txt')
        
        print(f"\nFold {i}:")
        print(f"  训练样本: {len(train_groups)} ({len(train_groups) * group_size} 行)")
        print(f"  验证样本: {len(valid_groups)} ({len(valid_groups) * group_size} 行)")
        print(f"  输出到: {fold_dir}/")
        
        write_groups_to_file(train_groups, train_file)
        write_groups_to_file(valid_groups, valid_file)
    
    # 验证
    print("\n✓ 验证切分正确性...")
    total_train = sum(len(train) for train, _ in folds)
    total_valid = sum(len(valid) for _, valid in folds)
    
    # 每个样本应该在某一个fold中恰好作为valid一次
    assert total_valid == n_samples, \
        f"验证集总数不匹配: {total_valid} != {n_samples}"
    # 总训练样本数应该是 n_samples * (k-1)
    expected_train = n_samples * (args.n_folds - 1)
    assert total_train == expected_train, \
        f"训练集总数不匹配: {total_train} != {expected_train}"
    
    print(f"✓ 所有 fold 生成成功！")
    print(f"\n输出目录: {output_base}/")


if __name__ == '__main__':
    main()
