import torch
import torch.nn as nn
import numpy as np
from .model_factory import register_model
from functools import partial

from cupy_layers.temporal_correlation import TemporalCorrelation
from cupy_layers.temporal_aggregation import TemporalAggregation

__all__ = ['C2D_DTF_ResNet', 'c2d_dtf_resnet50', 'c2d_dtf_resnet101']


def conv1x3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """1x3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=[1, 3, 3], stride=[1, stride, stride],
                     padding=[0, dilation, dilation], groups=groups, bias=False, dilation=[1, dilation, dilation])


def conv1x1x1(in_planes, out_planes, stride=1):
    """1x1x1 convolution"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=[1, stride, stride], bias=False)


class FFTLMul(nn.Module):
    def __init__(self, dim, shape, bias=True):
        super().__init__()
        self.dim = dim
        self.shape = shape
        self.bias = bias
    
    def forward(self, x, w):
        x = x * torch.view_as_complex(w)
        return x


class FFTLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.complex_weight = nn.Parameter(torch.randn([out_features, in_features, 2], dtype=torch.float32) * 0.02)
        if bias:
            bias_shape = [1, out_features, 1, 1, 1, 2]
            self.complex_bias = nn.Parameter(torch.zeros(bias_shape, dtype=torch.float32))

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias
        )

    def forward(self, x):
        B, C, T, H, W = x.shape
        w = torch.view_as_complex(self.complex_weight).unsqueeze(0).repeat(B, 1, 1)
        x = torch.bmm(w, x.reshape(B, C, T * H * W)).reshape(B, self.out_features, T, H, W)
        if self.bias:
            x = x + torch.view_as_complex(self.complex_bias)
        return x


class FFTBatchNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.norm = nn.BatchNorm3d(dim, affine=False)

    def extra_repr(self) -> str:
        return 'dim={}'.format(
            self.dim
        )

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = torch.view_as_real(x).reshape(B, C, T, H, W * 2)
        x = self.norm(x)
        x = x.reshape(B, C, T, H, W, 2)
        x = torch.view_as_complex(x)
        return x


class DTFBlock(nn.Module):
    def __init__(self, dim, clip_len=16, K=3, factor_G=16): 
        super().__init__()
        t = clip_len // 2 + 1
        self.clip_len = clip_len
        self.factor_G = factor_G

        # filter and spectrum learning 
        self.filter_g = nn.Conv2d(in_channels=(dim+K*K)*self.clip_len, out_channels=dim*t*2//self.factor_G, kernel_size=1, bias=False)
        self.filter1 = FFTLMul(dim=dim, shape=[t, 1, 1], bias=False)
        self.linear1 = FFTLinear(in_features=dim, out_features=dim, bias=False)
        self.norm1 = FFTBatchNorm(dim=dim)
        self.alpha1 = nn.Parameter(torch.zeros([1, dim, 1, 1, 1]))
        self.bias = nn.Parameter(torch.zeros([1, dim, 1, 1, 1]))

        # temporal correlation and frame-wise aggregation
        self.width = dim
        self.kernel_area, self.group_width = K*K, K*K
        self.cor_group_size,self.group_num, self.cor_dilation = dim, 1, 1
        self.pad_num = (self.cor_dilation * (K - 1) + 1) // 2
        self.K = K
        self._scale = torch.FloatTensor([1.0/(self.width)]).to('cuda')
        self.weights_cor = torch.nn.Parameter(torch.ones(self.width, clip_len, K, K).to('cuda')*self._scale, requires_grad=False)
        self.time_cor = TemporalCorrelation(self.width, self.group_width, kernel_size=K, 
                                            stride=1, padding=self.pad_num, dilation=self.cor_dilation, group_size=self.cor_group_size)   
        self.time_agg = TemporalAggregation(self.width, self.width*self.group_num, kernel_size=K, 
                                            stride=1, padding=self.pad_num, dilation=self.cor_dilation)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        B, C, T, H, W = x.shape
        t = T // 2 + 1
        K = self.K

        identity = x
        
        # -------------- DTF Block Forward ----------------
        # 1. ---- Frame-wise Aggregation ---
        # compute frame-wise correlation attention weights 
        w_o = self.time_cor(x, self.weights_cor)
        w_ = self.softmax(w_o)
        w_ = w_.transpose(1,2)
        w_ = w_.reshape(B*T, self.group_num, 1, self.kernel_area, H, W)
        # spatial aggregation
        a_ = self.time_agg(x.transpose(1,2).reshape(B*T, C, H, W), w_)
        a_ = a_.view(B, T, self.group_num*C, H, W).transpose(1,2)
        mask = torch.ones(x.size()).cuda()
        mask[:,:,-1,:,:] = 0
        mask.requires_grad = False
        a_shift = a_.clone()
        a_shift[:,:,:-1,:,:] = a_shift[:,:,1:,:,:]
        # temporal aggregation
        x = x + a_shift * mask 
        # ------------------------------
        # 2. ------- DTF operation --------
        # compute spectrum of input feature by FFT
        x_t = torch.fft.rfft(x, dim=2, norm='ortho')
        # concatenate input feature and correlation weights
        x_c = torch.cat([x, w_o], dim=1)
        # predict intermediate filter
        iterm_filter = self.filter_g(x_c.reshape(B, (C+K*K)*T, H, W))
        # expand intermediate filter to final filter
        final_filter = iterm_filter.unsqueeze(2).repeat(1, 1, self.factor_G, 1, 1)
        final_filter = final_filter.reshape(B, C, t, 2, H, W).permute(0,1,2,4,5,3).contiguous()
        # spectrum modulation
        y1 = self.norm1(self.linear1(self.filter1(x_t, final_filter)))
        # reconstruct visual feature by IFFT
        out = torch.fft.irfft(y1 * self.alpha1, n=T, dim=2, norm='ortho')
        # skip connection
        out = identity + out + self.bias
        # ------------------------------------------------

        return out


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x3x3(inplanes, planes, stride)
        self.conv1_blk = DTFBlock(dim=planes, clip_len=clip_length)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv1x3x3(planes, planes)
        self.conv2_blk = DTFBlock(dim=planes, clip_len=clip_length)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.conv1_blk(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.conv2_blk(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, clip_length=None, fft=True):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        width = int(planes * (base_width / 64.)) * groups
        self.fft = fft
        self.conv1 = conv1x1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv1x3x3(width, width, stride, groups, dilation)
        self.dtf_blk = DTFBlock(dim=width, clip_len=clip_length)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        # conv1
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # conv2
        out = self.conv2(out)
        # ------ DTF Block ------
        out = self.dtf_blk(out)
        # -----------------------
        out = self.bn2(out)
        out = self.relu(out)

        # conv3
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class C2D_DTF_ResNet(nn.Module):

    def __init__(self, block, layers, pooling_arch, early_stride=4, num_classes=1000, dropout_ratio=0.5,
                 zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, deep_stem=False, clip_length=16):
        super(C2D_DTF_ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d if not deep_stem else partial(nn.BatchNorm3d, eps=2e-5)
        self._norm_layer = norm_layer
        self._deep_stem = deep_stem

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        if not deep_stem:
            self.conv1 = nn.Conv3d(3, self.inplanes, kernel_size=[early_stride, 7, 7], stride=[early_stride, 2, 2],
                                   padding=[0, 3, 3],
                                   bias=False)
            self.bn1 = norm_layer(self.inplanes)
            self.relu = nn.ReLU(inplace=True)
        else:
            self.conv1 = nn.Conv3d(3, self.inplanes, kernel_size=[early_stride, 3, 3], stride=[early_stride, 2, 2],
                                   padding=[0, 1, 1],
                                   bias=False)
            self.bn1 = norm_layer(self.inplanes)
            self.relu1 = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv3d(self.inplanes, self.inplanes, kernel_size=[1, 3, 3], stride=[1, 1, 1], padding=[0, 1, 1],
                                   bias=False)
            self.bn2 = norm_layer(self.inplanes)
            self.relu2 = nn.ReLU(inplace=True)
            self.conv3 = nn.Conv3d(self.inplanes, self.inplanes * 2, kernel_size=[1, 3, 3], stride=[1, 1, 1], padding=[0, 1, 1],
                                   bias=False)
            self.bn3 = norm_layer(self.inplanes * 2)
            self.relu3 = nn.ReLU(inplace=True)

            self.inplanes *= 2
        self.maxpool = nn.MaxPool3d(kernel_size=[1, 3, 3], stride=[1, 2, 2], padding=[0, 1, 1])

        self.layer1 = self._make_layer(block, 64, layers[0], clip_length=clip_length//4)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0], clip_length=clip_length//4)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1], clip_length=clip_length//4)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2], clip_length=clip_length//4)
        
        self.inplanes = 256 * block.expansion

        self.pool = pooling_arch(input_dim=512 * block.expansion)
        self.drop = nn.Dropout(dropout_ratio)
        self.fc = nn.Linear(self.pool.output_dim, num_classes)

        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm, nn.SyncBatchNorm)):
                if m.affine:
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, clip_length=16, fft=True):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, clip_length = clip_length, fft=fft))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, clip_length = clip_length, fft=fft))

        return nn.Sequential(*layers)

    def _forward_impl(self, x, layer):
        bsz = x.size(0)

        # See note [TorchScript super()]
        if not self._deep_stem:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
        else:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu1(x)
            x = self.bn2(x)
            x = self.relu2(x)
            x = self.conv3(x)
            x = self.bn3(x)
            x = self.relu3(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.pool(x)
        x = self.drop(x)
        x = self.fc(x)
     
        return x

    def forward(self, x, layer=7):
        return self._forward_impl(x, layer)


def transfer_weights(state_dict, early_stride):
    new_state_dict = {}
    for k, v in state_dict.items():
        v = v.detach().numpy()
        if ('conv' in k) or ('downsample.0' in k):  # first conv7x7 layer
            shape = v.shape
            v = np.reshape(v, newshape=[shape[0], shape[1], 1, shape[2], shape[3]])
            if (not ('layer' in k)) and ('conv1' in k):  # first conv7x7 layer
                if early_stride != 1:
                    s1 = early_stride // 2
                    s2 = early_stride - early_stride // 2 - 1
                    v = np.concatenate((np.zeros(shape=(shape[0], shape[1], s1, shape[2], shape[3])), v,
                                        np.zeros(shape=(shape[0], shape[1], s2, shape[2], shape[3]))), axis=2)
        new_state_dict[k] = torch.from_numpy(v)
    return new_state_dict


def _c2d_dtf_resnet(arch, block, layers, pooling_arch, image_size=None, **kwargs):
    model = C2D_DTF_ResNet(block, layers, pooling_arch, **kwargs)
    return model


@register_model
def c2d_dtf_resnet50(pooling_arch, **kwargs):
    return _c2d_dtf_resnet('resnet50', Bottleneck, [3, 4, 6, 3], pooling_arch, **kwargs)


@register_model
def c2d_dtf_resnet101(pooling_arch, **kwargs):
    return _c2d_dtf_resnet('resnet101', Bottleneck, [3, 4, 23, 3], pooling_arch, **kwargs)

