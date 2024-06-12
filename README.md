# [TWC] Pay Less But Get More: A Dual-Attention-based Channel Estimation Network for Massive MIMO Systems with Low-Density Pilots
This is the official implementation of **[Pay Less But Get More: A Dual-Attention-based Channel Estimation Network for Massive MIMO Systems with Low-Density Pilots](https://ieeexplore.ieee.org/document/10315065)**, which has been published by **IEEE Transactions on Wireless Communications**.

To reap the promising benefits of massive multiple-input multiple-output (MIMO) systems, accurate channel state information (CSI) is required through channel estimation. However, due to the complicated wireless propagation environment and large-scale antenna arrays, precise channel estimation for massive MIMO systems is significantly challenging and costs an enormous training overhead. Considerable time-frequency resources are consumed to acquire sufficient accuracy of CSI, which thus severely degrades systems' spectral and energy efficiencies. In this paper, we propose a **dual-attention-based channel estimation network (DACEN)** to realize accurate channel estimation via low-density pilots, by jointly learning the spatial-temporal domain features of massive MIMO channels with the temporal attention module and the spatial attention module. To further improve the estimation accuracy, we propose a parameter-instance transfer learning approach to transfer the channel knowledge learned from the high-density pilots pre-acquired during the training dataset collection period. Experimental results reveal that the proposed DACEN-based method achieves better channel estimation performance than the existing methods under various pilot-density settings and signal-to-noise ratios. Additionally, with the proposed parameter-instance transfer learning approach, the DACEN-based method achieves additional performance gain, thereby further demonstrating the effectiveness and superiority of the proposed method.

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
@article{zhou2024pay,
  title = {Pay Less but Get More: A Dual-Attention-Based Channel Estimation Network for Massive {{MIMO}} Systems with Low-Density Pilots},
  author = {Zhou, Binggui and Yang, Xi and Ma, Shaodan and Gao, Feifei and Yang, Guanghua},
  journal = {IEEE Transactions on Wireless Communications},
  year = {2024},
  month = jun,
  volume = {23},
  number = {6},
  pages = {6061-6076},
  doi = {10.1109/TWC.2023.3329945}
}
```
