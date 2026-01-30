#!/usr/bin/env python
"""
验证 K-fold 数据切分的正确性

检查项：
1. 所有 fold 的 train + valid 样本数之和 = 原始 train 样本数
2. 各 fold 的 valid 集无重叠
3. 每个文件的行数能被 group_size 整除
"""

import os
import argparse
import tomlkit


def count_lines(filepath):
    """统计文件行数"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f)


def read_first_lines(filepath, n):
    """读取前 n 行（用于检查重叠）"""
    lines = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            lines.append(line)
    return ''.join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='验证 K-fold 数据切分的正确性'
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
        '--data_dir',
        default='data',
        help='数据目录（默认：data）'
    )
    parser.add_argument(
        '--config',
        default='data/datasets.toml',
        help='配置文件路径（默认：data/datasets.toml）'
    )
    
    args = parser.parse_args()
    
    # 读取配置
    with open(args.config, 'r', encoding='utf-8') as f:
        datasets = tomlkit.load(f)
    
    if args.dataset not in datasets:
        raise ValueError(f"数据集 '{args.dataset}' 未找到")
    
    config = datasets[args.dataset]
    inputs = config['inputs']
    group_size = len(inputs) + 1
    
    print(f"数据集: {args.dataset}")
    print(f"group_size: {group_size}")
    
    # 读取原始训练数据行数
    original_train = os.path.join(args.data_dir, config['train'])
    original_lines = count_lines(original_train)
    original_samples = original_lines // group_size
    
    print(f"\n原始训练数据:")
    print(f"  文件: {original_train}")
    print(f"  行数: {original_lines}")
    print(f"  样本数: {original_samples}")
    
    if original_lines % group_size != 0:
        print(f"  ⚠️  警告: 行数不能被 group_size 整除（余 {original_lines % group_size}）")
    
    # 检查每个 fold
    print(f"\n检查 {args.n_folds} 个 fold:")
    total_train_samples = 0
    total_valid_samples = 0
    
    valid_signatures = []  # 用于检查重叠
    
    for i in range(1, args.n_folds + 1):
        fold_dir = os.path.join(args.data_dir, args.dataset, 'folds', f'fold{i}')
        train_file = os.path.join(fold_dir, 'train.txt')
        valid_file = os.path.join(fold_dir, 'valid.txt')
        
        print(f"\n  Fold {i}:")
        
        # 检查文件存在
        if not os.path.exists(train_file):
            print(f"    ❌ 训练文件不存在: {train_file}")
            continue
        if not os.path.exists(valid_file):
            print(f"    ❌ 验证文件不存在: {valid_file}")
            continue
        
        # 统计行数和样本数
        train_lines = count_lines(train_file)
        valid_lines = count_lines(valid_file)
        
        train_samples = train_lines // group_size
        valid_samples = valid_lines // group_size
        
        print(f"    训练集: {train_samples} 样本 ({train_lines} 行)")
        print(f"    验证集: {valid_samples} 样本 ({valid_lines} 行)")
        
        # 检查行数能否被 group_size 整除
        if train_lines % group_size != 0:
            print(f"    ❌ 训练集行数不能整除 group_size（余 {train_lines % group_size}）")
        if valid_lines % group_size != 0:
            print(f"    ❌ 验证集行数不能整除 group_size（余 {valid_lines % group_size}）")
        
        total_train_samples += train_samples
        total_valid_samples += valid_samples
        
        # 读取验证集的前 group_size 行作为签名（用于检查重叠）
        valid_sig = read_first_lines(valid_file, group_size)
        valid_signatures.append((i, valid_sig))
    
    # 验证总样本数
    print(f"\n总结:")
    print(f"  所有 fold 训练样本数之和: {total_train_samples}")
    print(f"  所有 fold 验证样本数之和: {total_valid_samples}")
    print(f"  原始训练样本数: {original_samples}")
    
    expected_train_total = original_samples * (args.n_folds - 1)
    expected_valid_total = original_samples
    
    # 检查
    checks_passed = True
    
    if total_valid_samples != expected_valid_total:
        print(f"  ❌ 验证集总数不匹配！期望 {expected_valid_total}，实际 {total_valid_samples}")
        checks_passed = False
    else:
        print(f"  ✓ 验证集总数正确")
    
    if total_train_samples != expected_train_total:
        print(f"  ❌ 训练集总数不匹配！期望 {expected_train_total}，实际 {total_train_samples}")
        checks_passed = False
    else:
        print(f"  ✓ 训练集总数正确")
    
    # 检查验证集重叠
    print(f"\n检查验证集重叠:")
    overlap_found = False
    for i in range(len(valid_signatures)):
        for j in range(i + 1, len(valid_signatures)):
            fold_i, sig_i = valid_signatures[i]
            fold_j, sig_j = valid_signatures[j]
            if sig_i == sig_j:
                print(f"  ❌ Fold {fold_i} 和 Fold {fold_j} 的验证集可能有重叠！")
                overlap_found = True
                checks_passed = False
    
    if not overlap_found:
        print(f"  ✓ 各 fold 验证集无明显重叠（基于首个样本检查）")
    
    # 最终结论
    print(f"\n{'='*50}")
    if checks_passed:
        print("✓ 所有检查通过！数据切分正确。")
    else:
        print("❌ 发现问题，请检查数据切分过程。")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
