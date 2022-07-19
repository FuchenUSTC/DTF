# [ECCV 2022] Dynamic Temporal Filtering in Video Models

This repository includes the pure source code of DTF. The architecture DTF-Net and DTF-Transformer is implemented with python in PyTorch framework. And the operation temporal correlation/aggregation to construct Frame-wise Aggregation (FA) is programmed based on CuPy library.

# Update
* 2022.7.19: Source code of DTF operation and model configuration


# Contents:

* [Environment](#environment)
* [DTF CuPy Files](#dtf-cupy-files)
* [DTF-Net and DTF-Transformer](#dtf-net-and-dtf-transformer)
* [Citation](#citation)

## Environment
- Python 3.8.0
- PyTorch 1.8.1
- CUDA 11.1
- cuDNN 8.0
- CuPy 11.1

We have integrated the complete running environment for our implementation into a docker image and will release it in the future.

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