# [TWC] Pay Less But Get More: A Dual-Attention-based Channel Estimation Network for Massive MIMO Systems with Low-Density Pilots
This is the official implementation of **[Pay Less But Get More: A Dual-Attention-based Channel Estimation Network for Massive MIMO Systems with Low-Density Pilots](https://ieeexplore.ieee.org/document/10315065)**, which has been accepted by **IEEE Transactions on Wireless Communications**.

## Dataset

Simulation dataset generated with the 3GPP CDL channel model using the [Matlab 5G Toolbox](https://ww2.mathworks.cn/en/products/5g.html). Detailed system setup is referred to Table II of the paper.

## Code Usage

- **DACEN.py**: Module definition of the DACEN
- **trainer_from_scratch**: Training script to train the DACEN from scratch with low-density pilots
- **trainer_TL_source**: Training script to train the DACEN from scratch with high-density pilots; the trained DACEN is then used as the source model for parameter transfer
- **trainer_TL_target**: Training script to train the DACEN with the proposed parameter-instance transfer learning algorithm with low-density pilots (original data samples and generated samples with instance transfer)
- **utils**: Some utility functions

## Citation
If you use this code for your research, please cite our paper:
```
@article{zhou2023pay,
  title = {Pay Less but Get More: A Dual-Attention-Based Channel Estimation Network for Massive {{MIMO}} Systems with Low-Density Pilots},
  author = {Zhou, Binggui and Yang, Xi and Ma, Shaodan and Gao, Feifei and Yang, Guanghua},
  year = {2023},
  journal = {IEEE Transactions on Wireless Communications},
  pages = {1--1},
  issn = {1558-2248},
  doi = {10.1109/TWC.2023.3329945}
}
```