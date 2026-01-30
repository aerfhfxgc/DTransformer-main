"""
5-Fold 交叉验证训练脚本（无测试集污染）

核心设计：
1. 在 train 集内部按学生/序列（group）切分 5 折
2. 每折使用 4 份训练 + 1 份验证，进行 early stopping
3. 统计 5 折平均验证 AUC 用于超参选择
4. test 集完全隔离，只在 --final_train 模式下最终评估一次

使用方式：
    # 5-fold CV 超参搜索
    python scripts/train_cv.py -d assist09 -p --n_folds 5 -lr 0.001 -cl -o output/cv/
    
    # 最终训练（用全部 train 训练，test 评估一次）
    python scripts/train_cv.py -d assist09 -p --final_train -lr 0.001 -cl -o output/final/
"""

import os
import json
import sys
from argparse import ArgumentParser
from typing import Optional

import torch
import tomlkit
from tqdm import tqdm
import numpy as np

from DTransformer.data import KTData
from DTransformer.data_cv import KTDataCV
from DTransformer.eval import Evaluator

DATA_DIR = "data"

# 加载数据集配置
datasets = tomlkit.load(open(os.path.join(DATA_DIR, "datasets.toml")))

# ============================================================
# 命令行参数配置
# ============================================================
parser = ArgumentParser(description="5-Fold 交叉验证训练（无测试集污染）")

# 通用选项
parser.add_argument("--device", help="运行设备", default="cpu")
parser.add_argument("-bs", "--batch_size", help="训练批大小", default=8, type=int)
parser.add_argument("-tbs", "--test_batch_size", help="验证/测试批大小", default=64, type=int)

# 数据集选项
parser.add_argument(
    "-d", "--dataset",
    help="选择数据集",
    choices=datasets.keys(),
    required=True,
)
parser.add_argument(
    "-p", "--with_pid",
    help="使用题目ID（难度调制）",
    action="store_true"
)

# 模型选项
parser.add_argument("-m", "--model", help="选择模型（默认 DTransformer）")
parser.add_argument("--d_model", help="模型隐藏维度", type=int, default=128)
parser.add_argument("--n_layers", help="Transformer 层数", type=int, default=3)
parser.add_argument("--n_heads", help="注意力头数", type=int, default=8)
parser.add_argument("--n_know", help="知识参数维度", type=int, default=32)
parser.add_argument("--dropout", help="Dropout 率", type=float, default=0.2)
parser.add_argument("--proj", help="对比学习投影层", action="store_true")
parser.add_argument("--hard_neg", help="使用硬负样本", action="store_true")

# 训练选项
parser.add_argument("-n", "--n_epochs", help="最大训练轮数", type=int, default=100)
parser.add_argument(
    "-es", "--early_stop",
    help="无改进时提前停止的轮数",
    type=int,
    default=10,
)
parser.add_argument("-lr", "--learning_rate", help="学习率", type=float, default=1e-3)
parser.add_argument("-l2", help="L2 正则化", type=float, default=1e-5)
parser.add_argument("-cl", "--cl_loss", help="使用对比学习损失", action="store_true")
parser.add_argument("--lambda", help="对比学习损失权重", type=float, default=0.1, dest="lambda_cl")
parser.add_argument("--window", help="预测窗口", type=int, default=1)

# 交叉验证选项
parser.add_argument("--n_folds", help="交叉验证折数", type=int, default=5)
parser.add_argument("--seed", help="随机种子", type=int, default=42)
parser.add_argument(
    "--final_train",
    help="最终训练模式：用全部 train 训练，在 test 上评估一次",
    action="store_true"
)

# 输出选项
parser.add_argument("-o", "--output_dir", help="输出目录")
parser.add_argument("-f", "--from_file", help="从已有模型继续训练", default=None)


# ============================================================
# 模型创建函数
# ============================================================
def create_model(args, dataset):
    """根据参数创建模型实例"""
    if args.model == "DKT":
        from baselines.DKT import DKT
        return DKT(dataset["n_questions"], args.d_model)
    
    elif args.model == "DKVMN":
        from baselines.DKVMN import DKVMN
        return DKVMN(dataset["n_questions"], args.batch_size)
    
    elif args.model == "AKT":
        from baselines.AKT import AKT
        return AKT(
            dataset["n_questions"],
            dataset["n_pid"],
            d_model=args.d_model,
            n_heads=args.n_heads,
            dropout=args.dropout,
        )
    
    else:
        from DTransformer.model import DTransformer
        return DTransformer(
            dataset["n_questions"],
            dataset["n_pid"],
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            n_know=args.n_know,
            lambda_cl=args.lambda_cl,
            dropout=args.dropout,
            proj=args.proj,
            hard_neg=args.hard_neg,
            window=args.window,
        )


# ============================================================
# 单轮训练函数
# ============================================================
def train_epoch(model, train_data, optim, args, seq_len):
    """执行一轮训练"""
    model.train()
    it = tqdm(iter(train_data), desc="Training")
    total_loss = 0.0
    total_pred_loss = 0.0
    total_cl_loss = 0.0
    total_cnt = 0
    
    for batch in it:
        if args.with_pid:
            q, s, pid = batch.get("q", "s", "pid")
        else:
            q, s = batch.get("q", "s")
            pid = None if seq_len is None else [None] * len(q)
        
        if seq_len is None:
            q, s, pid = [q], [s], [pid]
        
        for q, s, pid in zip(q, s, pid):
            q = q.to(args.device)
            s = s.to(args.device)
            if pid is not None:
                pid = pid.to(args.device)
            
            if args.cl_loss:
                loss, pred_loss, cl_loss = model.get_cl_loss(q, s, pid)
            else:
                loss = model.get_loss(q, s, pid)
            
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            
            total_loss += loss.item()
            total_cnt += 1
            
            postfix = {"loss": total_loss / total_cnt}
            if args.cl_loss:
                total_pred_loss += pred_loss.item()
                total_cl_loss += cl_loss.item()
                postfix["pred_loss"] = total_pred_loss / total_cnt
                postfix["cl_loss"] = total_cl_loss / total_cnt
            it.set_postfix(postfix)
    
    return total_loss / max(total_cnt, 1)


# ============================================================
# 验证函数
# ============================================================
def validate(model, valid_data, args, seq_len):
    """在验证集上评估"""
    model.eval()
    evaluator = Evaluator()
    
    with torch.no_grad():
        it = tqdm(iter(valid_data), desc="Validating")
        for batch in it:
            if args.with_pid:
                q, s, pid = batch.get("q", "s", "pid")
            else:
                q, s = batch.get("q", "s")
                pid = None if seq_len is None else [None] * len(q)
            
            if seq_len is None:
                q, s, pid = [q], [s], [pid]
            
            for q, s, pid in zip(q, s, pid):
                q = q.to(args.device)
                s = s.to(args.device)
                if pid is not None:
                    pid = pid.to(args.device)
                y, *_ = model.predict(q, s, pid)
                evaluator.evaluate(s, torch.sigmoid(y))
    
    return evaluator.report()


# ============================================================
# 单折训练函数
# ============================================================
def train_fold(args, dataset, train_data, valid_data, fold_idx: Optional[int] = None):
    """
    训练单折模型。
    
    参数:
        args: 命令行参数
        dataset: 数据集配置
        train_data: 训练数据加载器
        valid_data: 验证数据加载器
        fold_idx: 折索引（None 表示最终训练）
        
    返回:
        best_epoch: 最佳轮次
        best_result: 最佳验证结果
        best_model_path: 最佳模型路径（如果保存了的话）
    """
    seq_len = dataset["seq_len"] if "seq_len" in dataset else None
    fold_str = f"Fold {fold_idx + 1}" if fold_idx is not None else "Final"
    
    print(f"\n{'='*60}")
    print(f"{fold_str} 开始训练")
    print(f"训练样本数: {len(train_data)}, 验证样本数: {len(valid_data)}")
    print(f"{'='*60}")
    
    # 创建模型
    model = create_model(args, dataset)
    if args.from_file:
        model.load_state_dict(torch.load(args.from_file, map_location=lambda s, _: s))
    
    optim = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.l2
    )
    model.to(args.device)
    
    # 训练循环
    best = {"auc": 0}
    best_epoch = 0
    best_model_path = None
    
    for epoch in range(1, args.n_epochs + 1):
        print(f"\n{fold_str} - Epoch {epoch}/{args.n_epochs}")
        
        # 训练
        train_loss = train_epoch(model, train_data, optim, args, seq_len)
        
        # 验证
        result = validate(model, valid_data, args, seq_len)
        print(f"验证结果: {result}")
        
        # 更新最佳结果
        if result["auc"] > best["auc"]:
            best = result
            best_epoch = epoch
            
            # 保存模型
            if args.output_dir:
                if fold_idx is not None:
                    model_path = os.path.join(
                        args.output_dir, f"fold{fold_idx + 1}-epoch{epoch:03d}-auc{result['auc']:.4f}.pt"
                    )
                else:
                    model_path = os.path.join(
                        args.output_dir, f"final-epoch{epoch:03d}-auc{result['auc']:.4f}.pt"
                    )
                print(f"保存模型: {model_path}")
                torch.save(model.state_dict(), model_path)
                best_model_path = model_path
        
        # Early stopping
        if args.early_stop > 0 and epoch - best_epoch > args.early_stop:
            print(f"连续 {args.early_stop} 轮无改进，提前停止")
            break
    
    print(f"\n{fold_str} 完成: 最佳 Epoch {best_epoch}, AUC {best['auc']:.4f}")
    return best_epoch, best, best_model_path


# ============================================================
# 5-Fold 交叉验证主函数
# ============================================================
def run_cv(args):
    """执行 5-fold 交叉验证"""
    dataset = datasets[args.dataset]
    data_path = os.path.join(DATA_DIR, dataset["train"])
    inputs = dataset["inputs"]
    seq_len = dataset["seq_len"] if "seq_len" in dataset else None
    
    print(f"\n{'#'*60}")
    print(f"# {args.n_folds}-Fold 交叉验证")
    print(f"# 数据集: {args.dataset}")
    print(f"# 训练文件: {data_path}")
    print(f"# 随机种子: {args.seed}")
    print(f"{'#'*60}")
    
    # 创建 fold 切分
    folds = KTDataCV.create_folds(data_path, inputs, n_folds=args.n_folds, seed=args.seed)
    
    print(f"\n样本切分:")
    for i, fold in enumerate(folds):
        print(f"  Fold {i + 1}: {len(fold)} 样本")
    
    # 准备输出目录
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        config_path = os.path.join(args.output_dir, "config.json")
        json.dump(vars(args), open(config_path, "w"), indent=2)
    
    # 训练每一折
    fold_results = []
    for fold_idx in range(args.n_folds):
        train_data, valid_data = KTDataCV.get_fold_loaders(
            data_path, inputs, folds, fold_idx,
            batch_size=args.batch_size,
            valid_batch_size=args.test_batch_size,
            seq_len=seq_len,
        )
        
        best_epoch, best_result, _ = train_fold(
            args, dataset, train_data, valid_data, fold_idx
        )
        fold_results.append(best_result)
    
    # 统计平均结果
    print(f"\n{'='*60}")
    print(f"{args.n_folds}-Fold 交叉验证结果汇总")
    print(f"{'='*60}")
    
    metrics = ["auc", "acc", "mae", "rmse"]
    avg_results = {}
    std_results = {}
    
    for metric in metrics:
        values = [r[metric] for r in fold_results]
        avg_results[metric] = np.mean(values)
        std_results[metric] = np.std(values)
    
    print("\n各折结果:")
    for i, result in enumerate(fold_results):
        print(f"  Fold {i + 1}: " + ", ".join(f"{k}={v:.4f}" for k, v in result.items()))
    
    print("\n平均结果 (mean ± std):")
    for metric in metrics:
        print(f"  {metric}: {avg_results[metric]:.4f} ± {std_results[metric]:.4f}")
    
    # 保存结果
    if args.output_dir:
        results_path = os.path.join(args.output_dir, "cv_results.json")
        results = {
            "args": vars(args),
            "fold_results": fold_results,
            "avg_results": avg_results,
            "std_results": std_results,
        }
        json.dump(results, open(results_path, "w"), indent=2)
        print(f"\n结果已保存: {results_path}")
    
    return avg_results, std_results


# ============================================================
# 最终训练主函数
# ============================================================
def run_final_train(args):
    """
    最终训练模式：
    1. 用全部 train 数据训练
    2. 在 test 上评估一次（唯一一次读取 test）
    """
    dataset = datasets[args.dataset]
    train_path = os.path.join(DATA_DIR, dataset["train"])
    test_path = os.path.join(DATA_DIR, dataset["test"])
    inputs = dataset["inputs"]
    seq_len = dataset["seq_len"] if "seq_len" in dataset else None
    
    print(f"\n{'#'*60}")
    print(f"# 最终训练模式")
    print(f"# 数据集: {args.dataset}")
    print(f"# 训练文件: {train_path}")
    print(f"# 测试文件: {test_path} (仅最终评估一次)")
    print(f"{'#'*60}")
    
    # 准备输出目录
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        config_path = os.path.join(args.output_dir, "config.json")
        json.dump(vars(args), open(config_path, "w"), indent=2)
    
    # 加载全部训练数据
    train_data = KTDataCV.get_full_loader(
        train_path, inputs,
        batch_size=args.batch_size,
        seq_len=seq_len,
        shuffle=True,
    )
    
    # 为了 early stopping，从训练集中切出一小部分作为验证
    # 注意：这里的验证集仍然来自 train，不是 test
    folds = KTDataCV.create_folds(train_path, inputs, n_folds=10, seed=args.seed)
    _, valid_data = KTDataCV.get_fold_loaders(
        train_path, inputs, folds, fold_idx=0,  # 用 10% 作为验证
        batch_size=args.batch_size,
        valid_batch_size=args.test_batch_size,
        seq_len=seq_len,
    )
    # 重新加载 90% 作为训练
    train_indices = []
    for i in range(1, 10):
        train_indices.extend(folds[i])
    train_data = KTDataCV(
        train_path, inputs, train_indices,
        batch_size=args.batch_size, seq_len=seq_len, shuffle=True
    )
    
    print(f"\n训练样本数: {len(train_data)} (90%)")
    print(f"验证样本数: {len(valid_data)} (10%, 用于 early stopping)")
    
    # 训练
    best_epoch, best_valid_result, best_model_path = train_fold(
        args, dataset, train_data, valid_data, fold_idx=None
    )
    
    # 在 test 上最终评估
    print(f"\n{'='*60}")
    print(f"在测试集上进行最终评估（唯一一次读取 test）")
    print(f"{'='*60}")
    
    # 加载最佳模型
    model = create_model(args, dataset)
    if best_model_path:
        model.load_state_dict(torch.load(best_model_path, map_location=lambda s, _: s))
    model.to(args.device)
    
    # 加载测试数据
    test_data = KTData(
        test_path, inputs,
        seq_len=seq_len,
        batch_size=args.test_batch_size,
    )
    
    # 评估
    test_result = validate(model, test_data, args, seq_len)
    
    print(f"\n{'='*60}")
    print(f"最终测试结果")
    print(f"{'='*60}")
    print(f"验证集最佳 (Epoch {best_epoch}): " + ", ".join(f"{k}={v:.4f}" for k, v in best_valid_result.items()))
    print(f"测试集结果: " + ", ".join(f"{k}={v:.4f}" for k, v in test_result.items()))
    
    # 保存结果
    if args.output_dir:
        results_path = os.path.join(args.output_dir, "final_results.json")
        results = {
            "args": vars(args),
            "best_epoch": best_epoch,
            "valid_result": best_valid_result,
            "test_result": test_result,
        }
        json.dump(results, open(results_path, "w"), indent=2)
        print(f"\n结果已保存: {results_path}")
    
    return test_result


# ============================================================
# 主入口
# ============================================================
if __name__ == "__main__":
    args = parser.parse_args()
    print(f"\n参数配置: {args}")
    
    if args.final_train:
        # 最终训练模式
        result = run_final_train(args)
    else:
        # 交叉验证模式
        avg_results, std_results = run_cv(args)
    
    print("\n完成!")

