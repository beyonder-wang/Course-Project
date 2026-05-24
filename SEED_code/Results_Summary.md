## Method: Pre-training with SimCLR-like process and MoE
*Config: backbone: EEGLSTM, dataset: MDD, lr=5e-4, alpha=0.1, epoch=50/30*
| method | Val ACC | best ACC |
| --- | --- | --- |
| baseline |  |  |
| none MoE | 0.8641 | 0.8750 (epoch 27) |
| with MoE | 0.8547 | 0.8641 (epoch 22) |

## Method: BCIC2A ATCNet optimization + ensemble
*Config: backbone: ATCNet (n_windows=5), dataset: BCIC2A, fold=1, epoch=120, optimizer: Adam*
| method | key settings | Val ACC / Seed Sweep Best | Ensemble |
| --- | --- | --- | --- |
| ATCNet baseline (prior) | lr=1e-3, 30ep | **66.20%** | — |
| +mixup+plateau | lr=1e-3, 120ep, mixup_alpha=0.2, plateau patience=15 | **68.06%** (seed 37, 97) | **71.76%** (3-model soft-voting) |

### AI's notes:
实施完成总结

  新增文件清单

  ┌─────────────────────────────┬───────────────────────────────────┬──────────────────────────────────────┐
  │            文件             │               作用                │                 依赖                 │
  ├─────────────────────────────┼───────────────────────────────────┼──────────────────────────────────────┤
  │ model/kan.py                │ KAN 层 (KANLinear + KANMLP)       │ 无                                   │
  ├─────────────────────────────┼───────────────────────────────────┼──────────────────────────────────────┤
  │ model/band_decomposition.py │ FFT 频带分解 (δ/θ/α/β/γ)          │ 无                                   │
  ├─────────────────────────────┼───────────────────────────────────┼──────────────────────────────────────┤
  │ model/multiband_loss.py     │ 多头 NT-Xent 损失 + Joint InfoNCE │ 无                                   │
  ├─────────────────────────────┼───────────────────────────────────┼──────────────────────────────────────┤
  │ model/mamba_model.py        │ EEGMamba backbone                 │ 无 (CPU=纯PyTorch, GPU推荐mamba-ssm) │
  ├─────────────────────────────┼───────────────────────────────────┼──────────────────────────────────────┤
  │ model/multiband_simclr.py   │ 多频带 SimCLR encoder (含MoE版)   │ band_decomposition, moe              │
  ├─────────────────────────────┼───────────────────────────────────┼──────────────────────────────────────┤
  │ pretrainer_multiband.py     │ MultiBandPretrainer 训练循环      │ multiband_loss, multiband_simclr     │
  ├─────────────────────────────┼───────────────────────────────────┼──────────────────────────────────────┤
  │ 3_run_multiband_simclr.py   │ 多频带 SimCLR CLI入口脚本         │ 上述全部                             │
  ├─────────────────────────────┼───────────────────────────────────┼──────────────────────────────────────┤
  │ research_survey.md          │ 调研资料汇总                      │ —                                    │
  └─────────────────────────────┴───────────────────────────────────┴──────────────────────────────────────┘

  改动的文件

  ┌───────────────────┬─────────────────────────────────────────────┐
  │       文件        │                    改动                     │
  ├───────────────────┼─────────────────────────────────────────────┤
  │ model/__init__.py │ 新增 EEGMamba 到 MODEL_DICT；导出全部新模块 │
  ├───────────────────┼─────────────────────────────────────────────┤
  │ CLAUDE.md         │ 新增模型/脚本文档和 Quick Commands          │
  └───────────────────┴─────────────────────────────────────────────┘

  使用方式

  # Mamba backbone (直接挂入现有训练脚本)
  python 0_run_train.py --dataset MDD --model EEGMamba --epochs 30 --device cpu

  # 多频带 SimCLR 预训练
  python 3_run_multiband_simclr.py --action pretrain --dataset MDD --epochs_pretrain 50

  # 多频带 SimCLR 预训练 + 微调
  python 3_run_multiband_simclr.py --action both --dataset MDD --epochs_pretrain 50 --epochs_finetune 30

  # 多频带 SimCLR + MoE
  python 3_run_multiband_simclr.py --action both --dataset MDD --use_moe --epochs_pretrain 50

  已知限制

  - EEGMamba 在 CPU 上较慢：纯 PyTorch 的序列扫描是 O(L) 的 Python for-loop。短序列 (200步) 可用，SLEEP (6000步)
  建议等有 GPU 或装 mamba-ssm 再跑。
  - 多频带 SimCLR 计算量更大：5 个频带 × 2 个增强视图 = 10× 的 LSTM forward，但预训练阶段无标签所以 batch 大，总体可控。


第三批实施完成总结

  新增文件

  ┌────────────────────────┬─────────────────────────────────────────────────────────────────────────────────────────┐
  │          文件          │                                          内容                                           │
  ├────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────┤
  │ model/kan_backbones.py │ 5 个 KAN 分类头变体：EEGLSTM_KAN, EEGGRU_KAN, EEGNet_KAN, EEGMamba_KAN, SimpleMLP_KAN   │
  ├────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────┤
  │ model/attention_eeg.py │ 3 个注意力增强 EEGNet：EEGNet_SE, EEGNet_SimAM, EEGNet_SimAM_SE + 可复用模块            │
  │                        │ SEBlock/SimAM/SpatialAttention1D                                                        │
  └────────────────────────┴─────────────────────────────────────────────────────────────────────────────────────────┘

  MODEL_DICT 现在共 14 个模型

  ┌────────┬─────────────────┬──────────────┬─────────────────────────────────┐
  │  类别  │      模型       │ 参数量 (MDD) │              说明               │
  ├────────┼─────────────────┼──────────────┼─────────────────────────────────┤
  │ 基线   │ SimpleLinear    │ 8K           │ —                               │
  ├────────┼─────────────────┼──────────────┼─────────────────────────────────┤
  │ 基线   │ SimpleMLP       │ 1.06M        │ —                               │
  ├────────┼─────────────────┼──────────────┼─────────────────────────────────┤
  │ 基线   │ EEGNet          │ 1.6K         │ 最轻量                          │
  ├────────┼─────────────────┼──────────────┼─────────────────────────────────┤
  │ 基线   │ EEGGRU          │ 44K          │ —                               │
  ├────────┼─────────────────┼──────────────┼─────────────────────────────────┤
  │ 基线   │ EEGLSTM         │ 152K         │ 当前最强 81-87%                 │
  ├────────┼─────────────────┼──────────────┼─────────────────────────────────┤
  │ 基线   │ EEGMamba        │ 90K          │ 比 LSTM 参数少 40%              │
  ├────────┼─────────────────┼──────────────┼─────────────────────────────────┤
  │ KAN头  │ EEGLSTM_KAN     │ 218K         │ LSTM + KAN 分类头               │
  ├────────┼─────────────────┼──────────────┼─────────────────────────────────┤
  │ KAN头  │ EEGGRU_KAN      │ 111K         │ GRU + KAN 分类头                │
  ├────────┼─────────────────┼──────────────┼─────────────────────────────────┤
  │ KAN头  │ EEGNet_KAN      │ 3.2K         │ EEGNet + KAN 分类头             │
  ├────────┼─────────────────┼──────────────┼─────────────────────────────────┤
  │ KAN头  │ EEGMamba_KAN    │ 123K         │ Mamba + KAN 分类头              │
  ├────────┼─────────────────┼──────────────┼─────────────────────────────────┤
  │ KAN头  │ SimpleMLP_KAN   │ 4.7M         │ 不推荐(输入维度过高致 KAN 膨胀) │
  ├────────┼─────────────────┼──────────────┼─────────────────────────────────┤
  │ 注意力 │ EEGNet_SE       │ 2.0K         │ +SE 通道注意力                  │
  ├────────┼─────────────────┼──────────────┼─────────────────────────────────┤
  │ 注意力 │ EEGNet_SimAM    │ 1.6K         │ +SimAM 无参注意力（参数不增）   │
  ├────────┼─────────────────┼──────────────┼─────────────────────────────────┤
  │ 注意力 │ EEGNet_SimAM_SE │ 2.0K         │ SE + SimAM 双注意力             │
  └────────┴─────────────────┴──────────────┴─────────────────────────────────┘

  使用方式

  所有新模型都是 backbone，直接用 0_run_train.py：

  # KAN 分类头
  python 0_run_train.py --dataset MDD --model EEGLSTM_KAN --epochs 30

  # 注意力 EEGNet（无参版）
  python 0_run_train.py --dataset MDD --model EEGNet_SimAM --epochs 30

  # 注意力 EEGNet（SE通道注意力）
  python 0_run_train.py --dataset MDD --model EEGNet_SE --epochs 30

  全项目现有文件结构

  model/
  ├── __init__.py           ← MODEL_DICT (14模型)
  ├── simple.py             ← SimpleLinear, SimpleMLP
  ├── eegnet.py             ← EEGNet
  ├── rnn.py                ← EEGGRU, EEGLSTM
  ├── mamba_model.py        ← EEGMamba          [NEW]
  ├── kan.py                ← KANLinear, KANMLP [NEW]
  ├── kan_backbones.py      ← 5个KAN分类头变体   [NEW]
  ├── attention_eeg.py      ← 3个注意力EEGNet    [NEW]
  ├── simclr_model.py       ← SimCLREncoder, MoESimCLREncoder
  ├── multiband_simclr.py   ← MultiBandSimCLR*  [NEW]
  ├── multiband_loss.py     ← MultiBandNTXentLoss [NEW]
  ├── band_decomposition.py ← BandDecomposition [NEW]
  ├── contrastive_loss.py   ← NTXentLoss
  ├── augmentations.py      ← 数据增强
  ├── channel_adapter.py    ← Phase 2适配器
  └── moe.py                ← MoELayer

  训练脚本:
  ├── 0_run_train.py              ← 监督训练(14模型)
  ├── 1_run_pretrain.py           ← 标准SimCLR预训练+微调
  ├── 2_run_benchmark.py          ← 多数据集benchmark
  └── 3_run_multiband_simclr.py   ← 多频带SimCLR [NEW]

  Pretrainer:
  ├── trainer.py                  ← 监督训练循环
  ├── pretrainer.py               ← 标准SimCLR训练循环
  └── pretrainer_multiband.py     ← 多频带SimCLR训练循环 [NEW]

  建议实验顺序：先跑 EEGNet_SimAM 和 EEGLSTM_KAN 看 baseline 对比，然后再跑 3_run_multiband_simclr.py 做多频带预训练。