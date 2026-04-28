# EEG 分类模型/架构调研 (2024–2025)

> 调研日期: 2026-04-28
> 目标: 为项目寻找提升模型性能的可行方案

---

## 一、Foundation Models（大规模预训练基础模型）

| 模型/论文 | 年份 | 核心思路 | 访问 |
|---|---|---|---|
| **LUNA** (NeurIPS 2025) | 2025 | Topology-agnostic, learned queries + cross-attention 统一不同通道数。21k+ 小时预训练, 300× FLOPs 降低 | ✅ 开放, [github.com/pulp-bio/biofoundation](https://github.com/pulp-bio/biofoundation) |
| **CSBrain** (Cross-scale Spatiotemporal Brain FM) | 2025 | 跨 11 个 EEG 任务、16 个数据集, 一致超过 task-specific SOTA | ✅ arXiv 开放 |
| **CBraMod** (Criss-Cross Brain FM) | 2025 | Parallel attention + asymmetric conditional positional encoding 分别建模时空 | ✅ arXiv 开放 |
| **EEG Foundation Models: A Survey** (Lai & Wei) | 2025.04 | 14 个 EEG FM 全面综述, 分析架构/预训练策略/下游任务 | ✅ `arxiv.org/abs/2504.20069` |
| **Self-supervised Learning for EEG: A Survey** (ACM Computing Surveys) | 2025 | SSL for EEG 全面综述 | ✅ 开放 |

---

## 二、Mamba (State-Space Models)

相比 Transformer O(n²) → O(n) 线性复杂度, 适合长序列 (SLEEP 30s×200Hz=6000 步)。

| 模型/论文 | 任务 | 精度 | 访问 |
|---|---|---|---|
| **DG-Mamba** (Mamba + Dynamic GNN) | 癫痫检测/睡眠分期 | AUROC 0.931, 10× 训练加速 | ✅ PubMed 开放 |
| **Mamba-CCA** (Mamba + Class Confusion-Aware Attention) | 情绪识别 (SEED) | 96.02% | ✅ eScholarship 开放 |
| **MSGM** (Multi-Scale Spatiotemporal Graph Mamba) | 情绪识别 | 边缘 151ms 推理 | ✅ arXiv 开放 |
| **SUMamba** (Mamba + Multi-scale fusion for SSVEP) | SSVEP BCI | SOTA | — |
| **HSF-BPM** (Hierarchical Spatial-Frequency Bi-Patch Mamba) | 情绪识别 | SOTA | ✅ [github.com/lxy185/HSF-BPM](https://github.com/lxy185/HSF-BPM) |

---

## 三、Transformer-based

| 模型/论文 | 任务 | 精度 | 访问 |
|---|---|---|---|
| **Large EEG-U-Transformer** (arXiv 2025) | 癫痫检测 — 2025 挑战赛冠军 | 冠军方案 | ✅ arXiv 开放 |
| **DB-STFFCNet** (Dual-Branch Spatio-Temporal-Frequency Fusion) | 运动想象 (BCI IV-2a) | 83.13% | ✅ MDPI 开放 |
| **ViT for iEEG** (Frontiers, NeuroPace) | 癫痫分类 136k iEEG | 96.8% | ✅ Frontiers 开放 |
| **SSTAF** (Spatial-Spectral-Temporal Attention Fusion Transformer) | 运动想象 | 76.83% | ✅ IEEE Access 开放 |

---

## 四、KAN (Kolmogorov-Arnold Networks)

可学习 B-spline 激活替代 MLP 固定激活。可替换项目中的 MLP 分类头/投影头。

| 模型/论文 | 任务 | 精度 | 访问 |
|---|---|---|---|
| **KSA-Mamba-PySPConv** (KAN + Mamba + Pyramidal Conv) | 运动想象 | 96.76% | ✅ Frontiers 开放 |
| **KAN-SleepNet** (KAN + BiLSTM) | 睡眠分期 | 85.1%, κ=0.792 | ✅ PMC, [github.com/xzhenliang/KAN-SleepNet](https://github.com/xzhenliang/KAN-SleepNet) |
| **KAleep-Net** (KAN + Flash Attention, IEEE TNSRE) | 睡眠分期 | 86.5% | — |
| **GRU+KAN** | 情绪识别 | 91.13%, **50% 更少参数** | — |
| **CA-KAN** (Convolutional Attention KAN) | 情绪识别 (SEED) | 94.63% | — |
| **KAN for Seizure Detection** | 癫痫检测 — 噪声下比 MLP 更稳定 | 稳定性分析 | ✅ DOAJ 开放 |

**推荐**: 用 KAN 替换 SimCLR projection head 或分类头中的 MLP。

---

## 五、EEGNet 变体

| 模型 | 改进点 | 精度 | 参数量 | 访问 |
|---|---|---|---|---|
| **LMA-EEGNet** (2024) | 空洞深度可分离卷积 + 多注意力双分支 | 95.71% | **2,471** (23KB) | ✅ MDPI 开放 |
| **EEGNet + SimAM Attention** | 无参注意力 SimAM | 72.9-87.3% | 极轻量 | — |
| **SincEEGNet** (2024) | 可解释频域滤波器 Sinc 卷积 | SOTA | 少参数 | ✅ 开放 |
| **ConvMixer-ECA** | ConvMixer + Efficient Channel Attention | 94.52% | — | ✅ MDPI 开放 |

---

## 六、SimCLR / 对比学习改进（对项目最直接）

| 模型/论文 | 改进 | 访问 |
|---|---|---|
| **DGNet** (ICLR 2026 submission) | **多频带多头 SimCLR**: 5 个 head 分别处理 delta/theta/alpha/beta/gamma, 相对标准 SimCLR 提升 **25.4%** | ✅ 代码 [anonymous.4open.science/r/iclr2026-7FE2](https://anonymous.4open.science/r/iclr2026-7FE2) |
| **Contrastive RL for EEG** (EMIL Lab @ ASU, 2024) | 通道重组合增加对比样本 + EEG 专用数据增强 | ✅ 开放 |
| **Improvised SimCLR for Emotion** | CNN+BiLSTM + SimCLR + FFT 频域特征 | ✅ 开放 |
| **Mitigating Spurious Features** (NeurIPS 2025) | 频谱正则化改善 SimCLR 中的虚假关联 | ✅ [github.com/NaghmehGh/SpuriousCorrelation_SSRL](https://github.com/NaghmehGh/SpuriousCorrelation_SSRL) |

**最推荐**: DGNet 的多频带多头 SimCLR, 直接替换 NTXentLoss 和 projection head。

---

## 七、通道自适应架构（针对 Phase 2 多数据集）

| 模型 | 方案 | 访问 |
|---|---|---|
| **Composable Channel-Adaptive Architecture** (IEEE JBHI 2025) | 单通道特征提取 + vector-symbolic fusion, 任意通道数 | ✅ PubMed 开放 |
| **EEG Adapter (EAD)** (arXiv:2505.23107) | 通道数无关的 adapter, zero-shot 泛化 | ✅ arXiv 开放 |
| **DIVER-0** (ICML 2025 Workshop) | 全通道等变 EEG foundation model | ✅ 开放 |

---

## 八、可参考的代码库

| 仓库 | 说明 |
|---|---|
| [pulp-bio/biofoundation](https://github.com/pulp-bio/biofoundation) | LUNA 官方实现 |
| [xzhenliang/KAN-SleepNet](https://github.com/xzhenliang/KAN-SleepNet) | KAN + BiLSTM 睡眠分期 |
| [lxy185/HSF-BPM](https://github.com/lxy185/HSF-BPM) | Mamba 情绪识别 |
| [anonymous.4open.science/r/iclr2026-7FE2](https://anonymous.4open.science/r/iclr2026-7FE2) | DGNet 多频带多头的 SimCLR |

---

## 九、推荐改进优先级

### Tier 1: 立刻可做（改代码少, 收益明确）
1. **多频带多头 SimCLR** — 参考 DGNet, 改造 NTXentLoss + projection head
2. **SimCLR projection head 替换为 KAN** — 替代 SimpleMLP

### Tier 2: 中等改动
3. **Mamba 替换 LSTM** — 引入 `mamba_ssm`, 替换 `model/rnn.py` 中的 GRU/LSTM
4. **EEGNet + Attention** — 引入 SimAM 或 SE-block

### Tier 3: 高收益长期
5. **迁移到 LUNA / EEG Foundation Model 框架**
6. **读综述选方向** — 先从 `arXiv:2504.20069` 开始
