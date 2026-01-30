"""
支持 K-Fold 交叉验证的数据加载模块。

关键设计：
1. 按 group（学生/序列）为单位切分，不拆散同一学生的数据
2. 支持按 fold 索引加载训练集/验证集
3. 与原有 KTData 接口兼容
"""

import linecache
import math
import subprocess
import sys
import random
from typing import List, Tuple, Optional

import torch
from torch.utils.data import DataLoader

from DTransformer.data import Batch, transform_batch


class LinesSubset:
    """
    Lines 的子集视图，通过索引列表访问原始文件的部分 group。
    
    与原 Lines 类接口兼容，但只访问 indices 指定的样本。
    """
    
    def __init__(self, filename: str, indices: List[int], group: int = 1, skip: int = 0):
        """
        参数:
            filename: 数据文件路径
            indices: 要访问的样本索引列表（基于 group 的索引）
            group: 每个样本占用的行数
            skip: 文件开头要跳过的行数
        """
        self.filename = filename
        self.indices = indices
        self.group = group
        self.skip = skip
        
        # 验证文件存在
        with open(filename):
            pass
    
    def __len__(self):
        return len(self.indices)
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
    
    def __getitem__(self, item):
        """
        支持整数索引和切片。
        """
        d = self.skip + 1
        
        if isinstance(item, int):
            if item < 0:
                item = len(self) + item
            if item < 0 or item >= len(self):
                raise IndexError(f"索引 {item} 超出范围 [0, {len(self)})")
            
            # 映射到原始文件的索引
            orig_idx = self.indices[item]
            
            if self.group == 1:
                line = linecache.getline(self.filename, orig_idx + d)
                return line.strip("\r\n")
            else:
                lines = [
                    linecache.getline(self.filename, d + orig_idx * self.group + k).strip("\r\n")
                    for k in range(self.group)
                ]
                return lines
        
        elif isinstance(item, slice):
            start = item.start or 0
            stop = item.stop or len(self)
            step = item.step or 1
            
            if start < 0:
                start = len(self) + start
            if stop < 0:
                stop = len(self) + stop
            
            start = max(0, min(start, len(self)))
            stop = max(0, min(stop, len(self)))
            
            return [self[i] for i in range(start, stop, step)]
        
        raise IndexError(f"不支持的索引类型: {type(item)}")


class KTDataCV:
    """
    支持 K-Fold 交叉验证的知识追踪数据加载器。
    
    用法:
        # 获取 5-fold 切分
        folds = KTDataCV.create_folds(data_path, inputs, n_folds=5, seed=42)
        
        # 加载第 i 折的训练/验证数据
        train_loader, valid_loader = KTDataCV.get_fold_loaders(
            data_path, inputs, folds, fold_idx=0,
            batch_size=8, valid_batch_size=64
        )
    """
    
    def __init__(
        self,
        data_path: str,
        inputs: List[str],
        indices: List[int],
        batch_size: int = 1,
        seq_len: Optional[int] = None,
        shuffle: bool = False,
        num_workers: int = 0,
    ):
        """
        参数:
            data_path: 数据文件路径
            inputs: 输入字段列表，如 ["pid", "q", "s"]
            indices: 要加载的样本索引列表
            batch_size: 批大小
            seq_len: 序列长度（用于长序列切分）
            shuffle: 是否打乱顺序
            num_workers: DataLoader 工作进程数
        """
        self.data = LinesSubset(data_path, indices, group=len(inputs) + 1)
        self.loader = DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=transform_batch,
            num_workers=num_workers,
        )
        self.inputs = inputs
        self.seq_len = seq_len
    
    def __iter__(self):
        return iter(self.loader)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return Batch(
            torch.tensor(
                [
                    [int(x) for x in line.strip().split(",")]
                    for line in self.data[index][1:]  # 跳过第一行（头信息）
                ]
            ),
            self.inputs,
            self.seq_len,
        )
    
    @staticmethod
    def count_samples(data_path: str, inputs: List[str]) -> int:
        """
        统计数据文件中的样本数量。
        
        参数:
            data_path: 数据文件路径
            inputs: 输入字段列表
            
        返回:
            样本数量
        """
        group = len(inputs) + 1
        
        # 计算文件行数
        if sys.platform == "win32":
            linecount = sum(1 for _ in open(data_path))
        else:
            output = subprocess.check_output(("wc -l " + data_path).split())
            linecount = int(output.split()[0])
        
        return linecount // group
    
    @staticmethod
    def create_folds(
        data_path: str,
        inputs: List[str],
        n_folds: int = 5,
        seed: int = 42,
    ) -> List[List[int]]:
        """
        创建 K-Fold 切分的索引列表。
        
        参数:
            data_path: 数据文件路径
            inputs: 输入字段列表
            n_folds: 折数
            seed: 随机种子（保证可复现性）
            
        返回:
            folds: List[List[int]]，每个元素是一折的样本索引列表
        """
        n_samples = KTDataCV.count_samples(data_path, inputs)
        
        # 生成并打乱索引
        indices = list(range(n_samples))
        rng = random.Random(seed)
        rng.shuffle(indices)
        
        # 切分为 n_folds 份
        fold_size = n_samples // n_folds
        remainder = n_samples % n_folds
        
        folds = []
        start = 0
        for i in range(n_folds):
            # 前 remainder 个 fold 多分配一个样本
            size = fold_size + (1 if i < remainder else 0)
            folds.append(indices[start:start + size])
            start += size
        
        return folds
    
    @staticmethod
    def get_fold_loaders(
        data_path: str,
        inputs: List[str],
        folds: List[List[int]],
        fold_idx: int,
        batch_size: int = 8,
        valid_batch_size: int = 64,
        seq_len: Optional[int] = None,
        num_workers: int = 0,
    ) -> Tuple['KTDataCV', 'KTDataCV']:
        """
        获取指定折的训练和验证数据加载器。
        
        参数:
            data_path: 数据文件路径
            inputs: 输入字段列表
            folds: 由 create_folds 生成的折索引列表
            fold_idx: 当前使用哪一折作为验证集（0 到 n_folds-1）
            batch_size: 训练批大小
            valid_batch_size: 验证批大小
            seq_len: 序列长度
            num_workers: DataLoader 工作进程数
            
        返回:
            (train_loader, valid_loader)
        """
        n_folds = len(folds)
        if fold_idx < 0 or fold_idx >= n_folds:
            raise ValueError(f"fold_idx 必须在 [0, {n_folds}) 范围内")
        
        # 验证集：第 fold_idx 折
        valid_indices = folds[fold_idx]
        
        # 训练集：其他所有折合并
        train_indices = []
        for i in range(n_folds):
            if i != fold_idx:
                train_indices.extend(folds[i])
        
        train_loader = KTDataCV(
            data_path, inputs, train_indices,
            batch_size=batch_size, seq_len=seq_len, shuffle=True, num_workers=num_workers
        )
        
        valid_loader = KTDataCV(
            data_path, inputs, valid_indices,
            batch_size=valid_batch_size, seq_len=seq_len, shuffle=False, num_workers=num_workers
        )
        
        return train_loader, valid_loader
    
    @staticmethod
    def get_full_loader(
        data_path: str,
        inputs: List[str],
        batch_size: int = 8,
        seq_len: Optional[int] = None,
        shuffle: bool = True,
        num_workers: int = 0,
    ) -> 'KTDataCV':
        """
        获取完整数据集的加载器（用于最终训练）。
        
        参数:
            data_path: 数据文件路径
            inputs: 输入字段列表
            batch_size: 批大小
            seq_len: 序列长度
            shuffle: 是否打乱
            num_workers: DataLoader 工作进程数
            
        返回:
            data_loader
        """
        n_samples = KTDataCV.count_samples(data_path, inputs)
        indices = list(range(n_samples))
        
        return KTDataCV(
            data_path, inputs, indices,
            batch_size=batch_size, seq_len=seq_len, shuffle=shuffle, num_workers=num_workers
        )

