## Method: Pre-training with SimCLR-like process and MoE
*Config: backbone: EEGLSTM, dataset: MDD, lr=5e-4, alpha=0.1, epoch=50/30*
| method | Val ACC | best ACC |
| --- | --- | --- |
| baseline |  |  |
| none MoE | 0.8641 | 0.8750 (epoch 27) |
| with MoE | 0.8547 | 0.8641 (epoch 22) |