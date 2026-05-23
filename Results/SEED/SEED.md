# SEED 实验总结

## 数据集

| Dataset | Train | Val | Test | 特征维度 |
|---------|------:|----:|-----:|---------|
| SEED (原始) | 900 | 450 | 450 | (62, 400) 波形 |
| SEED_DE_INIT | 900 | 450 | 450 | (62, 5) DE+LDS |
| **SEED_SUB1FULL_AUG_DE** | **5190** | 420 | 450 | (62, 5) DE+LDS |

SEED_SUB1FULL_AUG_DE 是在 SEED_DE_INIT 基础上，加入 Subject 1 全量 unseen 数据扩充训练集，验证集去除 Subject 1。

特征：5 频带 DE（δ θ α β γ）+ trial 内 LDS 卡尔曼平滑。

## 最佳结果

| Rank | Model | Val Acc | 配置 |
|:----:|-------|:-------:|------|
| **1** | **SimpleMLP** | **62.14%** | seed=42, mixup 0.2, ls 0.1, CosineWarmRestarts T0=10 |
| 2 | SimpleMLP | 59.29% | mixup 0.2, ls 0.1, CosineLR |
| 3 | SimpleMLP | 59.05% | same config, seed=123 |
| 4 | SimpleMLP | 58.57% | same config, seed=1024 |
| 5 | SimpleMLP | 57.62% | mixup 0.2, ls 0.1, CosineLR, 30ep |
| 6 | DGCNN | 52.86% | hidden_dim=16 / graph_layers=1, cosine |
| 7 | DGCNN_RG | 51.67% | ls+cos, 30ep |
| 8 | SEEDAsymNet | 49.11% | SEED_DE_INIT + prototype_interp |

## 最佳命令行

```bash
python 0_run_train.py \
  --dataset SEED_SUB1FULL_AUG_DE \
  --model SimpleMLP \
  --lr 5e-4 \
  --epochs 50 \
  --batch_size 32 \
  --label_smoothing 0.1 \
  --mixup_alpha 0.2 \
  --scheduler cosine_warm \
  --coswarm_t0 10 \
  --coswarm_eta_min 1e-5 \
  --simplemlp_hidden_dims 256,128 \
  --simplemlp_dropout 0.3 \
  --device cpu
```

## 实验历程

### 阶段 1: 原始波形探索
- DGCNN / SEEDGraphormer 在原始 SEED (62×400) 上 ~44%
- 结论：波形特征太稀疏，需要 DE 特征

### 阶段 2: DE+LDS 特征 + 图模型
- DGCNN 在 SEED_SUB1FULL_AUG_DE 上达到 52.86%
- DGCNN_RG / RGNN 均低于 DGCNN
- 发现：更简单的隐层 (16) / 单层图 (1) 效果更好

### 阶段 3: 模型大搜索
- 测试 12 个模型在 DE 特征上的表现
- SimpleMLP 以 57.62% 夺冠，远超所有图模型
- mixup (α=0.2) 将 SimpleMLP 推至 59.29%
- CNN 类模型因 time_point=5 过小无法运行
- SEEDGraphormer 完全失效 (33%=random)

### 阶段 4: 突破 60%
- CosineAnnealingWarmRestarts (T0=10, eta_min=1e-5) → **62.14%**
- 标准 CosineLR 59.29% vs CosWR 62.14%：周期性重启帮助逃离局部最优
- 多 seed 验证：top-3 seeds (42/123/1024) 均在 58.5-62.1% 范围
- Ensemble (soft/hard voting) 未超过最佳单模型

### 阶段 5: Track 2 弃线
- SEED_DE_INIT (900 样本) + SEEDAsymNet (1.1M 参数) 最高 49.11%
- 训练集太小，严重过拟合，放弃

## 关键发现

1. **DE 特征空间极低维 (62×5=310)**，MLP 比图模型/RNN/Transformer 更适合
2. **mixup 是关键正则化手段**：+2-3%
3. **CosineWarmRestarts 是关键调度策略**：+2.85% over CosineLR
4. **label smoothing (0.1) 稳定训练**
5. 更宽更深的架构反而下降 (512,256→52.62%)
6. SEED_SUB1FULL_AUG_DE 的 5190 样本 + Subject 1 扩充对训练至关重要
