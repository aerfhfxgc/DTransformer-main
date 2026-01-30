# 严格无测试集污染的实验流程指南

## 为什么需要此流程？

### 测试集污染问题

当前 DTransformer 的 `scripts/train.py` 存在一个严重问题：

```python
# scripts/train.py 第 91-98 行
valid_data = KTData(
    os.path.join(
        DATA_DIR, dataset["valid"] if "valid" in dataset else dataset["test"]  # ⚠️ 污染源
    ),
    ...
)
```

**问题**：当 `datasets.toml` 中没有 `valid` 字段时，训练脚本会使用 **test 集**作为验证集来：
- 进行每个 epoch 的验证
- 执行 early stopping
- 选择最佳模型

**后果**：这导致 **Test Set Contamination（测试集污染）**，使得最终在 test 上的评估结果**不再可信**，因为模型间接地在 test 上进行了超参调优。

### 正确做法

遵循标准机器学习实践：
1. **训练集（Train）**：用于模型参数学习
2. **验证集（Valid）**：用于超参调优和模型选择（从 train 内部切分）
3. **测试集（Test）**：**固定不变**，只在最终评估时使用**一次**

对于 K-fold 交叉验证：
- 在 **train 内部**按学生/序列切分 K 折
- 每折的 valid 用于该折的超参评估
- Test 始终保持独立

---

## 完整实验流程

### 准备工作

确保已安装必要的 Python 包：
```bash
pip install pandas tomlkit
```

### 步骤 1：生成 5-fold 数据切分

为每个数据集生成 5 折交叉验证数据：

```bash
# 为 assist09 生成 5-fold
python tools/make_folds.py -d assist09 -k 5 --seed 42

# 为其他数据集重复此操作
python tools/make_folds.py -d assist17 -k 5 --seed 42
python tools/make_folds.py -d algebra05 -k 5 --seed 42
python tools/make_folds.py -d statics -k 5 --seed 42
```

**输出结构**：
```
data/assist09/
├── train.txt          # 原始训练数据
├── test.txt           # 原始测试数据（保持不变）
└── folds/
    ├── fold1/
    │   ├── train.txt  # fold1 的训练集（约 80% 原始 train）
    │   └── valid.txt  # fold1 的验证集（约 20% 原始 train）
    ├── fold2/
    │   ├── train.txt
    │   └── valid.txt
    ... (fold3-5)
```

**重要说明**：
- 数据按 `group = len(inputs) + 1` 行为单位切分（保持学生序列完整性）
- 使用固定随机种子确保可复现
- 验证：各 fold 的 train + valid 样本数之和 = 原始 train 样本数

### 步骤 2：生成 Fold 配置文件

```bash
python tools/generate_folds_config.py \
    --input_config data/datasets.toml \
    --output_config data/datasets_folds.toml \
    --n_folds 5
```

这将生成 `data/datasets_folds.toml`，包含每个数据集的：
- `{dataset}_fold1` ~ `{dataset}_fold5`：5 个 fold 配置
- `{dataset}_final`：最终训练配置（稍后生成数据）

**配置示例**：
```toml
[assist09_fold1]
train = "assist09/folds/fold1/train.txt"
valid = "assist09/folds/fold1/valid.txt"  # ✅ 每个 fold 都有 valid
test = "assist09/test.txt"
n_questions = 123
n_pid = 17751
inputs = ["pid", "q", "s"]

[assist09_fold2]
# ... 类似结构
```

### 步骤 3：准备超参配置文件

创建 `hyperparams.json`（示例）：

```json
{
  "hyperparams": [
    {
      "d_model": 128,
      "n_layers": 3,
      "n_heads": 8,
      "n_know": 32,
      "dropout": 0.2,
      "lambda_cl": 0.1
    },
    {
      "d_model": 256,
      "n_layers": 3,
      "n_heads": 8,
      "n_know": 32,
      "dropout": 0.2,
      "lambda_cl": 0.1
    }
  ],
  "common_args": {
    "model": "DTransformer",
    "batch_size": 32,
    "test_batch_size": 64,
    "with_pid": true,
    "cl_loss": true,
    "proj": true
  }
}
```

### 步骤 4：运行 5-fold 交叉验证

**方式 A：自动化脚本（推荐）**

```bash
python tools/run_cross_validation.py \
    --dataset assist09 \
    --hyperparams_config hyperparams.json \
    --n_folds 5 \
    --output_dir cv_results \
    --config_file data/datasets_folds.toml \
    --max_epochs 100 \
    --early_stop 10 \
    --device cuda \
    --report_format both
```

这将：
1. 对每组超参配置运行 5 次训练（fold1-5）
2. 自动提取每折的 best valid AUC
3. 计算均值和标准差
4. 生成 CSV 和 Markdown 汇总报告
5. 标记出最佳配置

**输出**：
```
cv_results/
├── config_0/
│   ├── fold1/ (包含训练日志和模型)
│   ├── fold2/
│   ... fold3-5
├── config_1/
│   ... (同上)
├── cv_results.csv
└── cv_results.md
```

**方式 B：手动运行**

如果希望更精细地控制，可以手动运行每个 fold：

```bash
# Fold 1
python scripts/train.py \
    -d assist09_fold1 \
    --config data/datasets_folds.toml \
    -m DTransformer \
    -bs 32 -tbs 64 \
    -p -cl --proj \
    --d_model 256 --n_layers 3 --n_heads 8 --n_know 32 \
    --dropout 0.2 --lambda 0.1 \
    -n 100 -es 10 \
    -o output/cv/config_0/fold1 \
    --device cuda

# Fold 2-5：重复上述命令，修改 fold 编号和输出目录
```

然后手动汇总每折的 best valid AUC。

### 步骤 5：选择最佳超参

查看 `cv_results.csv` 或 `cv_results.md`：

```
config_id  d_model  ...  mean_auc  std_auc
0          128           0.7884    0.0032
1          256           0.7928    0.0021  ← 最佳配置
```

**选择标准**：
1. 首要：mean_auc 最高
2. 次要（同分时）：std_auc 最小（更稳定）

### 步骤 6：生成最终训练/验证切分

使用最佳超参训练最终模型前，需要从全量 train 切出 final_train/final_valid：

```bash
python tools/make_final_split.py \
    -d assist09 \
    --ratio 0.9 \
    --seed 42
```

**输出**：
```
data/assist09/final/
├── train.txt  # 90% 原始 train 样本
└── valid.txt  # 10% 原始 train 样本
```

**说明**：
- 最终模型仍需要 valid 用于 early stopping
- 推荐 90/10 切分（也可以根据数据量调整）
- `{dataset}_final` 配置已在步骤 2 自动生成

### 步骤 7：训练最终模型

使用步骤 5 选出的最佳超参训练最终模型：

```bash
python scripts/train.py \
    -d assist09_final \
    --config data/datasets_folds.toml \
    -m DTransformer \
    -bs 32 -tbs 64 \
    -p -cl --proj \
    --d_model 256 --n_layers 3 --n_heads 8 --n_know 32 \
    --dropout 0.2 --lambda 0.1 \
    -n 100 -es 10 \
    -o output/final_model \
    --device cuda
```

**记录**：
- 最佳模型文件：`output/final_model/model-XXX-0.XXXX.pt`
- 训练配置：`output/final_model/config.json`

### 步骤 8：最终评估（仅一次！）

**⚠️ 重要**：这是**第一次也是唯一一次**在 test 集上评估！

```bash
python scripts/test.py \
    -d assist09 \
    -m DTransformer \
    -bs 64 \
    -p \
    -f output/final_model/model-XXX-0.XXXX.pt \
    --device cuda
```

**输出**：test AUC 等指标，这才是最终可信的模型性能

---

## 防污染检查清单

在实验的各个阶段，使用此清单确保无测试集污染：

### ✅ 数据准备阶段
- [ ] 所有 fold 目录都只包含 `train.txt` 和 `valid.txt`
- [ ] `test.txt` 保持在原始位置，未被复制到 folds 目录
- [ ] 运行验证脚本确认切分正确性

### ✅ 配置文件检查
- [ ] 打开 `data/datasets_folds.toml`
- [ ] 每个 `{dataset}_fold{i}` 条目都包含：
  - `train` 字段 → 指向 fold 的 train.txt
  - `valid` 字段 → 指向 fold 的 valid.txt ✅
  - `test` 字段 → 指向原始 test.txt
- [ ] 每个 `{dataset}_final` 条目同样包含 train/valid/test 三个字段

### ✅ 交叉验证阶段
- [ ] 训练命令使用 `--config data/datasets_folds.toml`
- [ ] 训练命令使用 `-d {dataset}_fold{i}` 格式的数据集名
- [ ] 检查训练日志，确认：
  - 加载的 valid 路径是 `{dataset}/folds/fold{i}/valid.txt`
  - **没有**任何读取 `test.txt` 的记录
- [ ] 记录的 validation metrics 是基于 fold 的 valid 集

### ✅ 最终训练阶段
- [ ] 训练命令使用 `-d {dataset}_final`
- [ ] 加载的 valid 路径是 `{dataset}/final/valid.txt`
- [ ] Test 集**尚未被评估**

### ✅ 最终评估阶段
- [ ] 确认这是**第一次**运行 `scripts/test.py`
- [ ] 只运行**一次**测试
- [ ] 记录输出的 test metrics
- [ ] **不要**基于 test 结果再次调整超参或重新训练

---

## 工具命令参考

### make_folds.py

生成 K-fold 数据切分。

**选项**：
```
-d, --dataset TEXT       数据集名称（必需）
-k, --n_folds INT        fold 数量（默认：5）
--seed INT               随机种子（默认：42）
--data_dir TEXT          数据目录（默认：data）
--config TEXT            配置文件路径（默认：data/datasets.toml）
```

**示例**：
```bash
python tools/make_folds.py -d assist09 -k 5 --seed 42
python tools/make_folds.py -d assist17 -k 3 --seed 123  # 3-fold
```

### generate_folds_config.py

生成 datasets_folds.toml 配置文件。

**选项**：
```
--datasets [TEXT ...]    要生成配置的数据集列表（默认：所有）
--n_folds INT            fold 数量（默认：5）
--input_config TEXT      输入配置文件（默认：data/datasets.toml）
--output_config TEXT     输出配置文件（默认：data/datasets_folds.toml）
```

**示例**：
```bash
# 为所有数据集生成配置
python tools/generate_folds_config.py

# 仅为特定数据集生成
python tools/generate_folds_config.py --datasets assist09 assist17
```

### run_cross_validation.py

自动化运行交叉验证和超参搜索。

**选项**：
```
--dataset TEXT               数据集名称（必需）
--hyperparams_config TEXT    超参配置文件（JSON，必需）
--n_folds INT                fold 数量（默认：5）
--output_dir TEXT            输出目录（默认：cv_results）
--config_file TEXT           数据集配置文件（默认：data/datasets_folds.toml）
--max_epochs INT             最大训练轮数（默认：100）
--early_stop INT             early stop 轮数（默认：10）
--device TEXT                训练设备（默认：cpu）
--report_format TEXT         报告格式：csv|markdown|both（默认：both）
```

**示例**：
```bash
# 基本用法
python tools/run_cross_validation.py \
    --dataset assist09 \
    --hyperparams_config hyperparams.json

# 完整选项
python tools/run_cross_validation.py \
    --dataset assist09 \
    --hyperparams_config hyperparams.json \
    --n_folds 5 \
    --output_dir my_cv_results \
    --max_epochs 50 \
    --device cuda \
    --report_format markdown
```

### make_final_split.py

从全量 train 生成最终的 train/valid 切分。

**选项**：
```
-d, --dataset TEXT       数据集名称（必需）
--ratio FLOAT            训练集比例（默认：0.9）
--seed INT               随机种子（默认：42）
--data_dir TEXT          数据目录（默认：data）
--config TEXT            配置文件路径（默认：data/datasets.toml）
```

**示例**：
```bash
# 默认 90/10 切分
python tools/make_final_split.py -d assist09

# 自定义比例（80/20）
python tools/make_final_split.py -d assist09 --ratio 0.8
```

---

## 常见问题

### Q1: 为什么不直接用固定 epoch 数训练最终模型？

**A**: Early stopping 基于 valid 性能，不同超参配置可能在不同 epoch 收敛。固定 epoch 可能导致：
- 过拟合（epoch 过多）
- 欠拟合（epoch 不足）

使用 final_valid 进行 early stopping 更稳健。

### Q2: 5-fold 的 valid AUC 均值能代表最终 test 性能吗？

**A**: 不完全能。5-fold CV 的 valid AUC 均值用于：
- **相对比较**不同超参配置的性能
- 选择最佳配置

但由于：
- Train/test 数据分布可能不同
- 最终模型使用更多训练数据

最终 test AUC 可能与 CV 均值有差异（通常会更好）。

### Q3: 如果 test AUC 低于 CV 均值怎么办？

**A**: **不要**基于 test 结果重新调参！这样会引入 test 污染。正确做法：
1. 分析原因（数据分布差异、过拟合等）
2. 如果真的需要调整，**重新设计实验**（如增加 CV fold 数、调整验证策略）
3. 使用新的 test 集（如果可能）

### Q4: 可以用所有 train 数据训练最终模型吗（不留 valid）？

**A**: 可以，但需要**固定 epoch 数**（从 5-fold 的 best epoch 分布估算）。例如：
```bash
# 查看 CV 结果，假设最佳 epoch 集中在 40-50
python scripts/train.py \
    -d assist09 \
    --config data/datasets.toml \  # 使用原始配置（只有 train 和 test）
    <best_hyperparams> \
    -n 45 \  # 固定 epoch
    -es 0 \  # 禁用 early stopping
    -o output/final_model_full
```

**权衡**：
- ✅ 使用更多训练数据
- ❌ 失去 early stopping 保护，可能过拟合

推荐仍然使用 final_train/final_valid 方案。

### Q5: 数据切分时为什么要按 group 为单位？

**A**: 仓库的数据格式是：
```
<header line>
<field1 line>
<field2 line>
...
<fieldN line>
```

共 `group = len(inputs) + 1` 行组成一个样本。如果逐行随机切分会：
- 破坏样本完整性
- 导致数据损坏
- 可能泄露信息（同一学生的序列被切分到 train 和 valid）

---

## 总结

遵循此流程，你将获得:
- ✅ 无测试集污染的可信实验结果
- ✅ 标准的 5-fold 交叉验证超参选择
- ✅ 完整的可复现实验记录

**核心原则**：
1. **Test 固定不变**，只在最终评估一次
2. **所有调参/选择都在 train 内部完成**（通过 CV 的 valid）
3. **保持数据完整性**（按 group/学生切分）

祝实验顺利！
