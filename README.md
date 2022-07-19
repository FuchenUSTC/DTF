# [ECCV 2022] Dynamic Temporal Filtering in Video Models

## Note
This repository includes the pure source code files of DTF block and related network structures, i.e., DTF-Net and DTF-Transformer.

The network structures DTF-Net and DTF-Transformer are implemented with python in PyTorch framework, and the operation temporal correlation/aggregation to construct Frame-wise Aggregation (FA) is programmed based on CuPy library.

Because of the limitation of the package size, the related off-the-shelf pretrained models are not included in this repository. We will release them on Git in the future.

## Environment
- Python 3.8.0
- PyTorch 1.8.1
- CUDA 11.1
- cuDNN 8.0
- CuPy 11.1

We have integrated the complete running environment for our implementation into a docker image and will release it in the future.

## DTF CuPy Files
The source files of the temporal correlation and temporal aggregation to construct the Frame-wise Aggregation (FA) are in the folder `./cupy_dtf`.

Please correctly install the python library of CuPy (cupy-cuda111) to guarantee the success of the C++ code compilation.

You could check the gradient computation of the two operations with 
```
cd ./cupy_dtf && python temporal_correlation.py && python temporal_aggregation.py
```

## DTF-Net and DTF-Transformer Model Files
The related source files for the implementation of DTF-Net and DTF-Transformer are in the folder `./model_dtf`.

We define the network structure of DTF-Net in `./model_dtf/c2d_dtf_resnet.py` and the network structure of DTF-Transformer in `./model_dtf/c2d_dtf_swin.py`, respectively.

Please check them for more details.


## Code References
```
Python interface of temporal correlation operation: ./cupy_dtf/temporal_correlation.py line 275-306
Python interface of temporal aggregation operation: ./cupy_dtf/temporal_aggregation.py line 204-232
Frame-wise Aggregation (FA) operation in DTF block: ./model_dtf/c2d_dtf_resnet line 118-134
DTF block forward: ./model_dtf/c2d_dtf_resnet.py line line 117-151
DTF block after 3x3 conv in DTF-Net: ./model_dtf/c2d_dtf_resnet.py line 233-235
DTF block after W-MSA in DTF-Transformer: ./model_dtf/c2d_dtf_swin.py line 93-95
```