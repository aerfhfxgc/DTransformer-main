#!/usr/bin/env python
"""
自动化运行 K-fold 交叉验证并汇总结果

支持：
- 单组超参配置的 K-fold 验证
- 多组超参配置的网格搜索
- 自动提取 valid AUC 并计算统计量
- 生成 CSV/Markdown 汇总报告
"""

import os
import re
import json
import argparse
import subprocess
from pathlib import Path
import pandas as pd
from typing import Dict, List, Any


def parse_hyperparams_config(config_path: str) -> List[Dict[str, Any]]:
    """
    解析超参配置文件
    
    配置文件格式（JSON）：
    {
        "hyperparams": [
            {"d_model": 128, "n_layers": 3, "n_heads": 8, "dropout": 0.2},
            {"d_model": 256, "n_layers": 3, "n_heads": 8, "dropout": 0.2},
            ...
        ],
        "common_args": {
            "model": "DTransformer",
            "batch_size": 32,
            "with_pid": true,
            "cl_loss": true,
            "proj": true
        }
    }
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    return config.get('hyperparams', [{}]), config.get('common_args', {})


def build_train_command(
    dataset: str,
    fold: int,
    hyperparams: Dict[str, Any],
    common_args: Dict[str, Any],
    output_dir: str,
    config_file: str,
    max_epochs: int = 100,
    early_stop: int = 10
) -> List[str]:
    """
    构建训练命令
    
    Returns:
        命令参数列表
    """
    cmd = [
        'python', 'scripts/train.py',
        '-d', f'{dataset}_fold{fold}',
        '--config', config_file,
        '-n', str(max_epochs),
        '-es', str(early_stop),
        '-o', output_dir
    ]
    
    # 添加公共参数
    for key, value in common_args.items():
        if isinstance(value, bool):
            if value:
                if key == 'with_pid':
                    cmd.append('-p')
                elif key == 'cl_loss':
                    cmd.append('-cl')
                elif key == 'proj':
                    cmd.append('--proj')
                elif key == 'hard_neg':
                    cmd.append('--hard_neg')
        else:
            if key == 'model':
                cmd.extend(['-m', str(value)])
            elif key == 'batch_size':
                cmd.extend(['-bs', str(value)])
            elif key == 'test_batch_size':
                cmd.extend(['-tbs', str(value)])
    
    # 添加超参
    for key, value in hyperparams.items():
        cmd.append(f'--{key}')
        cmd.append(str(value))
    
    return cmd


def extract_best_valid_auc(output_dir: str) -> float:
    """
    从训练输出目录提取最佳 valid AUC
    
    查找规则：
    1. 读取 config.json（如果存在）
    2. 查找模型文件名中的 AUC 值（model-XXX-0.XXXX.pt）
    3. 返回最高的 AUC
    """
    if not os.path.exists(output_dir):
        raise FileNotFoundError(f"输出目录不存在: {output_dir}")
    
    # 查找所有模型文件
    model_files = list(Path(output_dir).glob('model-*.pt'))
    
    if not model_files:
        raise ValueError(f"在 {output_dir} 中未找到模型文件")
    
    # 从文件名提取 AUC
    # 格式：model-XXX-0.XXXX.pt
    auc_pattern = r'model-\d+-(\d+\.\d+)\.pt'
    
    best_auc = 0.0
    for model_file in model_files:
        match = re.search(auc_pattern, model_file.name)
        if match:
            auc = float(match.group(1))
            best_auc = max(best_auc, auc)
    
    if best_auc == 0.0:
        raise ValueError(f"无法从文件名提取 AUC 值: {output_dir}")
    
    return best_auc


def run_cross_validation(
    dataset: str,
    n_folds: int,
    hyperparams: Dict[str, Any],
    common_args: Dict[str, Any],
    config_id: int,
    base_output_dir: str,
    config_file: str,
    max_epochs: int,
    early_stop: int,
    device: str = 'cpu'
) -> Dict[str, Any]:
    """
    运行一组超参的 K-fold 交叉验证
    
    Returns:
        结果字典，包含每个 fold 的 AUC 和统计量
    """
    fold_results = []
    
    for fold in range(1, n_folds + 1):
        print(f"\n{'='*60}")
        print(f"Config {config_id} - Fold {fold}/{n_folds}")
        print(f"{'='*60}")
        
        # 构建输出目录
        output_dir = os.path.join(
            base_output_dir,
            f'config_{config_id}',
            f'fold{fold}'
        )
        
        # 构建训练命令
        cmd = build_train_command(
            dataset=dataset,
            fold=fold,
            hyperparams=hyperparams,
            common_args=common_args,
            output_dir=output_dir,
            config_file=config_file,
            max_epochs=max_epochs,
            early_stop=early_stop
        )
        
        # 添加设备参数
        if device != 'cpu':
            cmd.extend(['--device', device])
        
        print(f"命令: {' '.join(cmd)}")
        
        # 运行训练
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                encoding='utf-8'
            )
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"训练失败: {e}")
            print(f"错误输出: {e.stderr}")
            raise
        
        # 提取 valid AUC
        try:
            auc = extract_best_valid_auc(output_dir)
            fold_results.append(auc)
            print(f"\n✓ Fold {fold} 最佳 Valid AUC: {auc:.4f}")
        except Exception as e:
            print(f"提取 AUC 失败: {e}")
            raise
    
    # 计算统计量
    mean_auc = sum(fold_results) / len(fold_results)
    std_auc = (sum((x - mean_auc) ** 2 for x in fold_results) / len(fold_results)) ** 0.5
    
    result = {
        'config_id': config_id,
        **hyperparams,
        **{f'fold{i}_auc': auc for i, auc in enumerate(fold_results, 1)},
        'mean_auc': mean_auc,
        'std_auc': std_auc
    }
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description='运行 K-fold 交叉验证和超参搜索'
    )
    parser.add_argument(
        '--dataset',
        required=True,
        help='数据集名称（如 assist09）'
    )
    parser.add_argument(
        '--hyperparams_config',
        required=True,
        help='超参配置文件路径（JSON 格式）'
    )
    parser.add_argument(
        '--n_folds',
        type=int,
        default=5,
        help='fold 数量（默认：5）'
    )
    parser.add_argument(
        '--output_dir',
        default='cv_results',
        help='输出目录（默认：cv_results）'
    )
    parser.add_argument(
        '--config_file',
        default='data/datasets_folds.toml',
        help='数据集配置文件（默认：data/datasets_folds.toml）'
    )
    parser.add_argument(
        '--max_epochs',
        type=int,
        default=100,
        help='最大训练轮数（默认：100）'
    )
    parser.add_argument(
        '--early_stop',
        type=int,
        default=10,
        help='early stop 轮数（默认：10）'
    )
    parser.add_argument(
        '--device',
        default='cpu',
        help='训练设备（默认：cpu）'
    )
    parser.add_argument(
        '--report_format',
        choices=['csv', 'markdown', 'both'],
        default='both',
        help='报告格式（默认：both）'
    )
    
    args = parser.parse_args()
    
    # 解析超参配置
    print(f"读取超参配置: {args.hyperparams_config}")
    hyperparams_list, common_args = parse_hyperparams_config(args.hyperparams_config)
    
    print(f"\n总共 {len(hyperparams_list)} 组超参配置")
    print(f"公共参数: {common_args}")
    
    # 运行交叉验证
    all_results = []
    
    for config_id, hyperparams in enumerate(hyperparams_list):
        print(f"\n{'#'*60}")
        print(f"配置 {config_id}: {hyperparams}")
        print(f"{'#'*60}")
        
        result = run_cross_validation(
            dataset=args.dataset,
            n_folds=args.n_folds,
            hyperparams=hyperparams,
            common_args=common_args,
            config_id=config_id,
            base_output_dir=args.output_dir,
            config_file=args.config_file,
            max_epochs=args.max_epochs,
            early_stop=args.early_stop,
            device=args.device
        )
        
        all_results.append(result)
        
        print(f"\n配置 {config_id} 汇总:")
        print(f"  Mean AUC: {result['mean_auc']:.4f} ± {result['std_auc']:.4f}")
    
    # 生成报告
    df = pd.DataFrame(all_results)
    
    # 找出最佳配置
    best_idx = df['mean_auc'].idxmax()
    best_config = df.loc[best_idx]
    
    print(f"\n{'='*60}")
    print("最佳配置:")
    print(f"  Config ID: {best_config['config_id']}")
    print(f"  Mean AUC: {best_config['mean_auc']:.4f} ± {best_config['std_auc']:.4f}")
    print(f"  超参: ")
    for key, value in best_config.items():
        if key not in ['config_id', 'mean_auc', 'std_auc'] and not key.startswith('fold'):
            print(f"    {key}: {value}")
    print(f"{'='*60}")
    
    # 保存结果
    if args.report_format in ['csv', 'both']:
        csv_path = os.path.join(args.output_dir, 'cv_results.csv')
        df.to_csv(csv_path, index=False)
        print(f"\n✓ CSV 报告已保存: {csv_path}")
    
    if args.report_format in ['markdown', 'both']:
        md_path = os.path.join(args.output_dir, 'cv_results.md')
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write("# 交叉验证结果\n\n")
            f.write(df.to_markdown(index=False))
            f.write(f"\n\n## 最佳配置\n\n")
            f.write(f"- **Config ID**: {best_config['config_id']}\n")
            f.write(f"- **Mean AUC**: {best_config['mean_auc']:.4f} ± {best_config['std_auc']:.4f}\n")
            f.write(f"- **超参**: \n")
            for key, value in best_config.items():
                if key not in ['config_id', 'mean_auc', 'std_auc'] and not key.startswith('fold'):
                    f.write(f"  - {key}: {value}\n")
        print(f"✓ Markdown 报告已保存: {md_path}")


if __name__ == '__main__':
    main()
