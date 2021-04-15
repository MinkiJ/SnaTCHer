# Few-shot Open-set Recognition by Transformation Consistency [CVPR 2021]
Link to paper: [arXiv](https://arxiv.org/abs/2103.01537)

## Description
This code executes SnaTCHer-F on miniImageNet and tieredImageNet

## Prerequisites
* Ubuntu 18.04
* Python 3.7
* PyTorch 1.6
* CUDA 10.1
* Anaconda 4.9.2
* CUDNN v7.6.3

## Usage
1. Prepare datasets and checkpoints from [link](https://github.com/Sha-Lab/FEAT)
2. Move datasets under './data' (see main.py)
3. Move checkpoints under './checkpoints' with prefix (*mini-* or *tiered-*, see main.py)
4. Run main.py

### Running example
* miniImageNet 5-way 1-shot
```python main.py --dataset MiniImageNet --shot 1```

* tieredImageNet 5-way 5-shot
```python main.py --dataset TieredImageNet --shot 5```

## Results
|	| Acc (%)  | Probability (%) | Distance (%) | SnaTCHer (%) |
|:------|:---------:|:----------------:|:-------------:|:-------------:|
|Mini1shot| 66.15 | 59.37 | 68.74 | **69.39** |
|Mini5shot| 81.87 | 62.71 | 76.01 | **77.36** |
|Tiered1shot| 70.41 | 63.88 | 69.80 | **74.38** |
|Tiered5shot| 84.79 | 73.79 | 77.25 | **81.78** |

## SnaTCHer details
See model/trainer/fsl_trainer_SnatCHerF.py


## Acknowledgement
The code is based on [github.com/Sha-Lab/FEAT](https://github.com/Sha-Lab/FEAT)



