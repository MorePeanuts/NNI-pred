# NNI-pred 研究方案与方法说明

本文档详细说明项目的研究方法、数据处理流程和建模策略。

## 1. 研究概述

### 1.1 研究目标

本项目旨在预测环境样本（土壤和水体）中新烟碱类杀虫剂（NNI）的浓度分布。通过机器学习方法，探索影响污染物浓度的关键驱动因子，并建立可靠的预测模型。

### 1.2 目标污染物

**水体目标变量**：
- 母体污染物：THIA（噻虫嗪）、IMI（吡虫啉）、CLO（噻虫胺）、ACE（啶虫脒）、DIN（呋虫胺）、parentNNIs（母体总和）
- 代谢物：IMI-UREA、DN-IMI、DM-ACE、CLO-UREA、mNNIs（代谢物总和）

**土壤目标变量**：
- 母体污染物：THIA、IMI、CLO、parentNNIs
- 代谢物：IMI-UREA、DM-CLO、DN-IMI、CLO-UREA、mNNIs

## 2. 数据集结构

### 2.1 原始数据

项目包含两个独立采样的数据集：

| 数据集 | 样本量 | 空间位置数 | 时间维度 |
|--------|--------|------------|----------|
| 土壤   | 186    | 62         | 3 季节   |
| 水体   | 159    | 53         | 3 季节   |

> **注意**：水土采样点空间不重合，无时间序列数据，仅有季节分类（Dry、Normal、Rainy）。

### 2.2 特征分组

代码中将特征划分为以下几组（见 `src/nni_pred/data.py`）：

| 特征组 | 水体数据集 | 土壤数据集 | 说明 |
|--------|-----------|-----------|------|
| `group_natural` | 7 个 | 12 个 | 自然环境因子（气象、理化性质、植被等） |
| `group_agro` | 18 个 | 12 个 | 农业集约化特征（投入品、种植规模、产出等） |
| `group_socio` | 16 个 | 14 个 | 社会经济特征（GDP、城市化、用水等） |
| `categorical` | 2 个 | 2 个 | 分类变量（Season、Landuse） |

## 3. 数据预处理

### 3.1 水土样本匹配（IDW 空间融合）

对于水体数据集，通过反距离加权（IDW）方法融合周边土壤样本的特征（见 `scripts/idw_merge.py`）：

**匹配规则**：
1. 以水体样本为中心，搜索半径 R = 30 km
2. 仅匹配同一季节的土壤样本
3. 使用 Haversine 公式计算地球表面距离

**特征聚合方法**：

数值变量使用 IDW2 加权：

$$V_{soil\_agg} = \frac{\sum_{j=1}^{k} w_{ij} \cdot V_{soil\_j}}{\sum_{j=1}^{k} w_{ij}}$$

其中权重 $w_{ij} = \frac{1}{d_{ij}^2}$

分类变量（如 landuse）取最近邻土壤样本的值。

**聚合后的特征命名**：
- 数值特征：`Soil_{原特征名}_agg`（如 `Soil_THIA_agg`）
- 分类特征：`Soil_{原特征名}`（如 `Soil_landuse`）

### 3.2 分组 PCA 降维

为消除共线性同时保留物理意义，对 `group_agro` 和 `group_socio` 分别进行 PCA（见 `src/nni_pred/transformers.py`）：

```python
GroupedPCA(variance_threshold=0.95)
```

**处理流程**：
1. 分组标准化（StandardScaler）
2. 分组 PCA，保留 95% 方差
3. 输出主成分命名为 `PC_Agro_1, PC_Agro_2, ...` 和 `PC_Socio_1, PC_Socio_2, ...`

> **设计理由**：避免全局 PCA 将农业面源压力与社会经济压力混合。

### 3.3 目标变量变换

使用对数变换处理右偏的污染物浓度分布（见 `TargetTransformer` 类）：

$$y_{transformed} = \log(y + offset)$$

其中 `offset = min(positive_values) / 2`，在训练集上自动计算以处理零值。

### 3.4 线性模型专用处理

针对 Elastic Net 模型，还需额外处理：

1. **偏态矫正**：对偏度 > 0.75 的数值特征执行 `log1p` 变换
2. **特征缩放**：对自然特征组使用 Z-score 标准化
3. **分类编码**：使用 One-Hot 编码（drop='first' 避免共线性）

> **注意**：树模型（RF、XGBoost）使用 Ordinal 编码，无需额外缩放。

## 4. 建模方法

### 4.1 候选模型

| 模型 | 类型 | 特点 |
|------|------|------|
| Elastic Net | 线性 | 结合 L1 和 L2 正则化，可解释性强 |
| Random Forest | 集成 | 对异常值稳健，可处理非线性 |
| XGBoost | 梯度提升 | 高预测精度，支持 Tweedie 损失 |

### 4.2 嵌套交叉验证

采用空间分组的嵌套交叉验证（见 `src/nni_pred/trainer.py`）：

```
外层 CV（5-fold）：评估泛化性能
  └── 内层 CV（4-fold）：超参数调优
```

**关键设计**：
- 使用 `GroupKFold` 按空间位置分组，确保同一位置的三季数据不同时出现在训练集和测试集
- 外层评估使用 OOF（Out-of-Fold）预测，避免信息泄露

### 4.3 超参数优化

#### 4.3.1 搜索策略

采用网格搜索（`GridSearchCV`）进行超参数优化：
- **搜索方式**：穷举所有超参数组合
- **评分指标**：R²（决定系数），在内层 CV 上计算
- **最优选择**：选择内层 CV 平均 R² 最高的参数组合

#### 4.3.2 超参数空间

**Elastic Net**（36 组合）：

| 参数 | 说明 | 搜索范围 |
|------|------|----------|
| `alpha` | 正则化强度 | [0.0005, 0.001, 0.01, 0.1, 1.0, 10.0] |
| `l1_ratio` | L1/L2 混合比例 | [0.1, 0.3, 0.5, 0.7, 0.9, 0.99] |
| `max_iter` | 最大迭代次数 | [10000] |

> `l1_ratio=1` 等价于 Lasso，`l1_ratio=0` 等价于 Ridge

**Random Forest**（72 组合）：

| 参数 | 说明 | 搜索范围 |
|------|------|----------|
| `n_estimators` | 树的数量 | [100, 200] |
| `max_depth` | 最大深度 | [8, 15, 20] |
| `min_samples_split` | 分裂最小样本数 | [5, 8, 10] |
| `min_samples_leaf` | 叶节点最小样本数 | [2, 4] |
| `max_features` | 特征选择策略 | ['sqrt', 'log2'] |

**XGBoost**（64 组合）：

| 参数 | 说明 | 搜索范围 |
|------|------|----------|
| `n_estimators` | 提升迭代次数 | [100, 200] |
| `max_depth` | 树的最大深度 | [3, 5] |
| `learning_rate` | 学习率 | [0.05, 0.1] |
| `subsample` | 样本采样比例 | [0.8, 1.0] |
| `colsample_bytree` | 特征采样比例 | [0.8, 1.0] |
| `reg_alpha` | L1 正则化 | [0, 0.1] |
| `reg_lambda` | L2 正则化 | [1, 5] |
| `tweedie_variance_power` | Tweedie 方差幂 | [1.2, 1.8] |

> Tweedie 损失适用于右偏且含零值的数据，`power=1.2` 接近 Poisson，`power=1.8` 接近 Gamma

#### 4.3.3 最终模型训练

超参数选择完成后，使用最优参数在全数据集上重新进行网格搜索训练，以充分利用所有数据提升模型性能。

### 4.4 随机种子选择

通过多次实验自动选择最优随机种子（见 `SeedSelector` 类）：

1. 生成 N 个随机种子（默认 10 个）
2. 每个种子运行完整嵌套 CV
3. 计算变异系数（CV）筛选稳定结果：`CV = std / mean`
4. 选择指标最优的种子和模型

**筛选条件**：
- 变异系数阈值：默认 0.8（可配置）
- 指标选择：默认 `NSE_log`

## 5. 评估指标

### 5.1 核心指标

| 指标 | 公式 | 说明 |
|------|------|------|
| NSE | $1 - \frac{\sum(y - \hat{y})^2}{\sum(y - \bar{y})^2}$ | Nash-Sutcliffe 效率系数 |
| RSR | $\frac{RMSE}{std(y)}$ | RMSE 标准差比 |
| PBIAS | $\frac{\sum(y - \hat{y})}{\sum y} \times 100\%$ | 百分比偏差 |
| KGE | $1 - \sqrt{(r-1)^2 + (\alpha-1)^2 + (\beta-1)^2}$ | Kling-Gupta 效率 |

### 5.2 指标变体

- `NSE_log`：对数空间的 NSE（**默认选择指标**）
- `NSE_detected`：仅计算检出样本（浓度 > 检出限）
- 其他指标同理

### 5.3 评价等级

| 等级 | NSE/KGE | RSR | PBIAS |
|------|---------|-----|-------|
| Very Good | > 0.7 | 0-0.5 | < ±25% |
| Good | 0.55-0.7 | 0.5-0.6 | ±25%-±40% |
| Satisfactory | 0.4-0.55 | 0.6-0.7 | ±40%-±70% |
| Unsatisfactory | < 0.4 | > 0.7 | > ±70% |

### 5.4 检出率指标

- TPR（True Positive Rate）：正确预测检出的比例
- FNR（False Negative Rate）：漏检率
- FPR（False Positive Rate）：误报率
- TNR（True Negative Rate）：正确预测未检出的比例

## 6. 使用指南

### 6.1 运行完整实验

```bash
# 推荐命令（包含种子选择）
uv run scripts/train_all.py --max-attempts 5 --size medium

# 指定目标和数据类型
uv run scripts/train_all.py --cls soil --targets THIA IMI CLO

# 调整变异系数阈值
uv run scripts/train_all.py --cv-threshold 0.5 --indicator NSE_log
```

### 6.2 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--cls` | merged | 数据集类型：`merged`（水体+土壤特征）或 `soil` |
| `--size` | medium | 超参数网格规模：`small`、`medium`、`large` |
| `--max-attempts` | 10 | 随机种子尝试次数 |
| `--cv-threshold` | 0.8 | 变异系数筛选阈值 |
| `--indicator` | NSE_log | 模型选择指标 |
| `--targets` | all | 目标污染物列表 |

### 6.3 输出结构

```
output/exp_{cls}_{seed}_{timestamp}/
├── {target}/
│   ├── seed_{n}/
│   │   ├── oof_metrics_of_{model_type}.json  # 各模型 OOF 指标
│   │   ├── model_comparison.json              # 模型比较结果
│   │   └── {model_type}_model_for_{target}.joblib  # 最优模型
│   └── seed_comparison.json                   # 种子比较结果
└── metrics_summary.csv                        # 所有目标汇总
```

## 7. 注意事项

### 7.1 数据相关

1. **空间自相关**：必须使用 GroupKFold 按位置分组，否则会高估模型性能
2. **季节匹配**：IDW 融合时严格按季节匹配，确保水土时间同步性
3. **检出限处理**：低于检出限的样本使用检出限值的一半作为替代值

### 7.2 模型相关

1. **特征工程差异**：线性模型和树模型使用不同的预处理 Pipeline
2. **PCA 拟合时机**：PCA 仅在训练集上 fit，避免测试集信息泄露
3. **模型保存**：最终模型在全数据集上重新训练后保存

### 7.3 评估相关

1. **指标选择**：`NSE_log` 对低浓度预测更敏感，推荐作为主指标
2. **稳定性检验**：通过变异系数筛选，剔除不稳定的实验结果
3. **检出样本评估**：`_detected` 后缀指标仅在真正检出的样本上计算

## 8. 代码架构

```
src/nni_pred/
├── data.py          # 数据集类和特征分组定义
├── models.py        # 模型构建器（ElasticNet、RF、XGBoost）
├── transformers.py  # 特征工程 Pipeline（PCA、偏态处理等）
├── trainer.py       # 训练器和种子选择器
├── evaluation.py    # 评估指标和模型比较
├── visualization.py # 可视化（指标图、散点图、SHAP）
└── utils.py         # 实验结果探索工具

scripts/
├── train_all.py     # 完整实验脚本（含种子选择）
├── train_simplest.py # 单模型快速训练（调试用）
├── idw_merge.py     # 水土数据 IDW 融合
├── visualize.py     # 生成可视化图表
└── select_seed.py   # 独立种子选择脚本
```
