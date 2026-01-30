#!/usr/bin/env python
"""
从全量训练数据生成最终的 train/valid 切分

用于在选定最佳超参后，训练最终模型时使用。
推荐比例：90% train, 10% valid
"""

import os
import argparse
import random
import tomlkit


def read_groups_from_file(filepath, group_size):
    """按 group 读取数据文件"""
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
        )
    
    return groups


def write_groups_to_file(groups, filepath):
    """将样本块写入文件"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        for group in groups:
            for line in group:
                f.write(line)


def main():
    parser = argparse.ArgumentParser(
        description='生成最终训练的 train/valid 切分'
    )
    parser.add_argument(
        '-d', '--dataset',
        required=True,
        help='数据集名称（如 assist09）'
    )
    parser.add_argument(
        '--ratio',
        type=float,
        default=0.9,
        help='训练集比例（默认：0.9，即 90/10 切分）'
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
    print(f"切分比例: {args.ratio:.1%} train / {1-args.ratio:.1%} valid")
    
    # 读取原始训练数据
    train_file = os.path.join(args.data_dir, dataset_config['train'])
    print(f"\n读取训练数据: {train_file}")
    
    groups = read_groups_from_file(train_file, group_size)
    n_samples = len(groups)
    
    print(f"总样本块数: {n_samples}")
    print(f"总行数: {n_samples * group_size}")
    
    # 打乱并切分
    random.seed(args.seed)
    shuffled_groups = groups.copy()
    random.shuffle(shuffled_groups)
    
    split_point = int(n_samples * args.ratio)
    train_groups = shuffled_groups[:split_point]
    valid_groups = shuffled_groups[split_point:]
    
    print(f"\n切分结果 (seed={args.seed}):")
    print(f"  训练样本: {len(train_groups)} ({len(train_groups) * group_size} 行)")
    print(f"  验证样本: {len(valid_groups)} ({len(valid_groups) * group_size} 行)")
    
    # 输出
    output_dir = os.path.join(args.data_dir, args.dataset, 'final')
    train_output = os.path.join(output_dir, 'train.txt')
    valid_output = os.path.join(output_dir, 'valid.txt')
    
    print(f"\n输出到: {output_dir}/")
    write_groups_to_file(train_groups, train_output)
    write_groups_to_file(valid_groups, valid_output)
    
    print("✓ 最终切分生成成功！")


if __name__ == '__main__':
    main()
