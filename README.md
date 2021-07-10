# Orientation Field Estimation for Latent Fingerprints with Prior Knowledge of Fingerprint Pattern

> [Orientation Field Estimation for Latent Fingerprints with Prior Knowledge of Fingerprint Pattern](TODO:)  
> Yongjie Duan, Jianjiang Feng, Jiwen Lu, and Jie Zhou  
> IJCB 2021 [ [paper](TODO:), [highlight](TODO:), [presentation](TODO:) ]

We provide the clean-version training and evaluation code of our IJCB 2021 paper ["Orientation Field Estimation for Latent Fingerprints with Prior Knowledge of Fingerprint Pattern"](TODO:).

If you find this study useful for your reesearch, please cite our paper:

<!-- ```
@inproceedings{duan2021prior,
author = {Yongjie Duan, Jianjiang Feng, Jiwen Lu, and Jie Zhou},
booktitle = {Proceedings of the IEEE International Joint Conference on Biometrics},
pages = {TODO:},
title = {Orientation Field Estimation for Latent Fingerprints with Prior Knowledge of Fingerprint Pattern},
year = {2021}
}
``` -->

## Overview
stimating orientation field for latent fingerprints plays a crucial role in latent fingerprints recognition systems. Considering the intrinsic characteristics of fingerprints that the distribution of orientation field varies with the fingerprint patterns, we propose an orientation field estimation algorithm for latent fingerprints based on residual learning using prior knowledge of fingerprint patterns. Specifically, statistical distribution models of orientation field, for different fingerprint patterns, are calculated based on a large database consisting of 14,000 fingerprints with good quality using clustering method. The residual orientation fields and reliability scores, indicating the consistency with different statistical orientation models, are estimated using a deep network, named RefNet. Then the final orientation field is obtained by fusing the estimations according to their corresponding reliability scores.

![finger-pattern]{./images/patterns.png}  
![overview]{./images/overview.png}

## Enviroment
Set up Python 3.7.0 enviroment:
```
pytorch==1.7.0
torchvision==0.8.1
tensorboard==2.4.1
```

## Trained Model
Our trained model ([download link](https://drive.google.com/file/d/1u0aA8f9FV7jlE7QfOY_g2WdcJ1ZtpqVE/view?usp=sharing)).
Please download and unzip to the current directory as follows:

```
Root
├── checkpoints
│   └── multiresanchor_mixhard_20201005
│       ├── checkpoint
│       ├── config.json
│       └── epoch_79_.pth.tar
```

## Results
We conducted experiments on gerprint database NIST SD27 to compare the propose algorithm with state-of-the-art orientation field estimation algorithms.  
The accuracy of orientation field estimation is evaluated by the average Root Mean Square Deviation (RMSD)

| Methods      | All       | Good     | Bad       | Ugly      |
| ------------ | --------- | -------- | --------- | --------- |
| STFT         | 32.51     | 27.27    | 34.10     | 36.36     |
| FOMFE        | 28.12     | 22.83    | 29.09     | 32.63     |
| GlobalDict   | 18.44     | 14.40    | 19.18     | 21.88     |
| LocalDict    | 14.35     | 11.15    | 15.15     | 16.85     |
| LocalDict-M  | 13.76     | 10.87    | 14.12     | 16.40     |
| ConvNet      | 13.51     | 10.76    | 13.94     | 16.00     |
| SparseCoding | 16.38     | 12.57    | 16.88     | 20.22     |
| ExSearch-B   | 13.54     | 11.21    | 14.20     | 14.95     |
| ExSearch     | 13.01     | 10.85    | 13.99     | 14.27     |
| FingerNet    | 17.82     | 13.67    | 18.42     | 21.50     |
| Single       | 12.46     | 9.91     | 12.91     | 14.65     |
| Single-M     | 12.20     | 9.65     | 12.73     | 14.39     |
| Ours         | **12.16** | **9.87** | **12.83** | **13.85** |
| Ours-M       | **12.10** | **9.60** | **12.71** | **14.07** |

Fingerprint matching experiment is applied to examine the contribution of the proposed orientation field estimation in overall fingerprint identification system.
![cmc_nist27]{./images/cmc_nist27.png}
> CMC curves of different algorithms on latent fingerprint database NIST SD27 as well as three subsets: (a) all (258 latents), (b) good quality (88 latents), (c) bad quality (85 latents), (d) ugly quality (85 latents). “-M” denotes manually marked fingerprint pose.
