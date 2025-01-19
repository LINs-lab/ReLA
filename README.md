<p align="center">
<img src="./assets/poster.jpg" width=100% height=100% 
class="center">
</p>

<div align="center">
  <a href="https://sp12138.github.io/">Peng Sun :man_artist:</a>, Yi Jiang :man_student:, <a href="https://tlin-taolin.github.io/">Tao Lin :skier:</a>

  <a href="https://arxiv.org/abs/2405.14669">[arXiv] :page_facing_up:</a> | <a href="#bibliography">[BibTeX] :label:</a>
</div>


## Abstract
Data, the seminal opportunity and challenge in modern machine learning, currently constrains the scalability of representation learning and impedes the pace of model evolution.
In this work, we investigate the efficiency properties of data from both optimization and generalization perspectives.
Our theoretical and empirical analysis reveals an unexpected finding: for a given task, utilizing a publicly available, task- and architecture-agnostic model (referred to as the `prior model' in this paper) can effectively produce efficient data.
Building on this insight, we propose the Representation Learning Accelerator (ReLA), which promotes the formation and utilization of efficient data, thereby accelerating representation learning.
Utilizing a ResNet-18 pre-trained on CIFAR-10 as a prior model to inform ResNet-50 training on ImageNet-1K reduces computational costs by $50\%$ while maintaining the same accuracy as the model trained with the original BYOL, which requires $100\%$ cost.

## Usage

### Requirements

```
torchvision
torch
lightly
pytorch-lightning
```

### How to Run

The primary entry point for a single experiment is [`main.py`](main.py). To simplify the execution of multiple experiments, we provide a set of [`scripts`](scripts/) designed for running the bulk experiments detailed in the paper. For instance, to execute `ReLA` for accelerating the training of ResNet-18 with BYOL on the CIFAR-10 dataset, you can use the following command:

```shell
bash ./scripts/run.sh
```

### Storage Format for Raw Datasets

All our raw datasets, including those like ImageNet-1K and CIFAR10, store their training and validation components in the following format to facilitate uniform reading using a standard dataset class method:

```
/path/to/dataset/
├── 00000/
│   ├── image1.jpg
│   ├── image2.jpg
│   ├── image3.jpg
│   ├── image4.jpg
│   └── image5.jpg
├── 00001/
│   ├── image1.jpg
│   ├── image2.jpg
│   ├── image3.jpg
│   ├── image4.jpg
│   └── image5.jpg
├── 00002/
│   ├── image1.jpg
│   ├── image2.jpg
│   ├── image3.jpg
│   ├── image4.jpg
│   └── image5.jpg
```

This organizational structure ensures compatibility with the unified dataset class, streamlining the process of data handling and accessibility.

## Bibliography

If you find this repository helpful for your project, please consider citing our work:

```
@inproceedings{
sun2024efficiency,
title={Efficiency for Free: Ideal Data Are Transportable Representations},
author={Peng Sun and Yi Jiang and Tao Lin},
booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
year={2024},
url={https://openreview.net/forum?id=UPxmISfNCO}
}
```

## Reference

Our code has referred to previous work: [LightlySSL](https://github.com/lightly-ai/lightly)