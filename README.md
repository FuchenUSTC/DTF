# [ECCV 2022] Dynamic Temporal Filtering in Video Models

This repository includes the pure source code of DTF. The architecture DTF-Net and DTF-Transformer is implemented with python in PyTorch framework. And the operation temporal correlation/aggregation to construct Frame-wise Aggregation (FA) is programmed based on CuPy library.

# Update
* 2022.7.19: Source code of DTF operation and model configuration


# Contents:
* [Paper Introduction](#paper-introduction)
* [Environment](#environment)
* [Base Repository of DTF](#base-repository-of-dtf)
* [DTF CuPy Files](#dtf-cupy-files)
* [DTF-Net and DTF-Transformer](#dtf-net-and-dtf-transformer)
* [Citation](#citation)


# Paper Introduction
<div align=center>
<img src="https://raw.githubusercontent.com/FuchenUSTC/DTF/master/pic/DTF.PNG" width="600" alt="image"/>
</div>

Video temporal dynamics is conventionally modeled with 3D spatial-temporal kernel or its factorized version comprised of 2D spatial kernel and 1D temporal kernel. The modeling power, nevertheless, is limited by the fixed window size and static weights of a kernel along the temporal dimension. The pre-determined kernel size severely limits the temporal receptive fields and the fixed weights treat each spatial location across frames equally, resulting in sub-optimal solution for long-range temporal modeling in natural scenes. In this paper, we present a new recipe of temporal feature learning, namely Dynamic Temporal Filter (DTF), that novelly performs spatial-aware temporal modeling in frequency domain with large temporal receptive field. Specifically, DTF dynamically learns a specialized frequency filter for every spatial location to model its long-range temporal dynamics. Meanwhile, the temporal feature of each spatial location is also transformed into frequency feature spectrum via 1D Fast Fourier Transform (FFT). The spectrum is modulated by the learnt frequency filter, and then transformed back to temporal domain with inverse FFT. In addition, to facilitate the learning of frequency filter in DTF, we perform frame-wise aggregation to enhance the primary temporal feature with its temporal neighbors by inter-frame correlation. It is feasible to plug DTF block into ConvNets and Transformer, yielding DTF-Net and DTF-Transformer. Extensive experiments conducted on three datasets demonstrate the superiority of our proposals. More remarkably, DTF-Transformer achieves an accuracy of 83.5% on Kinetics-400 dataset.


# Environment
- Python 3.8.0
- PyTorch 1.8.1
- CUDA 11.1
- cuDNN 8.0
- CuPy 11.1

We have integrated the complete running environment for our implementation into a docker image and will release it in the future.


# Base Repository of DTF

The data processing, training and evaluation pipeline of DTF is totally based on the repository of our another work [SIFA](https://github.com/FuchenUSTC/SIFA). It is readily to integrate the DTF networks into the SIFA training pipeline by drawing the related files into the folder `./model` and modifing `./model/__init__.py` in SIFA repository. 

# DTF CuPy Files
The source files of the temporal correlation and temporal aggregation to construct the Frame-wise Aggregation (FA) are in the folder `./cupy_dtf`.

Please correctly install the python library of CuPy (cupy-cuda111) to guarantee the success of the C++ code compilation.

You could check the gradient computation of the two operations with 
```
cd ./cupy_dtf && python temporal_correlation.py && python temporal_aggregation.py
```

# DTF-Net and DTF-Transformer
The related source files for the implementation of DTF-Net and DTF-Transformer are in the folder `./model_dtf`.

We define the network structure of DTF-Net in `./model_dtf/c2d_dtf_resnet.py` and the network structure of DTF-Transformer in `./model_dtf/c2d_dtf_swin.py`, respectively.

Please check them for more details.


# Citation

If you use these models in your research, please cite:

    @inproceedings{Long:ECCV22,
      title={Dynamic Temporal Filtering in Video Models},
      author={Fuchen Long, Zhaofan Qiu, Yingwei Pan, Ting Yao, Chong-Wah Ngo and Tao Mei},
      booktitle={ECCV},
      year={2022}
    }