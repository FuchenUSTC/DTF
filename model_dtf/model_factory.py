import sys
import numpy as np
import torch
from skimage import transform

model_dict = {}
transfer_dict = {}


def get_model_by_name(net_name, **kwargs):
    return model_dict.get(net_name)(**kwargs)


def transfer_weights(net_name, state_dict, early_stride=4):
    if transfer_dict[net_name] is None:
        raise NotImplementedError
    else:
        return transfer_dict[net_name](state_dict, early_stride)


def remove_fc(net_name, state_dict):
    if net_name.startswith('c2d_eftnet'):
        state_dict.pop('_fc.weight', None)
        state_dict.pop('_fc.bias', None)
        return state_dict
    state_dict.pop('fc.weight', None)
    state_dict.pop('fc.bias', None)
    if net_name.startswith('lgd3d_') or net_name.startswith('lgd_p3d_'):
        state_dict.pop('fc_g.weight', None)
        state_dict.pop('fc_g.bias', None)
    if net_name.startswith('dg_p3da_') or net_name.startswith('dg_p3d_') or net_name.startswith('dgtc_c2d_'):
        state_dict.pop('fc_dual.weight', None)
        state_dict.pop('fc_dual.bias', None)
    return state_dict


def convert_fft_weight(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        v = v.detach().numpy()
        if 'weights_cor' in k or 'freq_' in k: continue
        if 'filter2.complex_weight' in k:
            shape = v.shape # [1, dim, t, 1, ,1, 2]
            v = np.reshape(v, newshape=[shape[1], shape[2], 2])
            tn = 2 * shape[2] - 1
            if 'dual' in k: tn = shape[2] + 1
            if tn != shape[2]:
                v1 = transform.resize(v[:,:,0], (shape[1], tn)).reshape(shape[1], tn, 1)
                v2 = transform.resize(v[:,:,1], (shape[1], tn)).reshape(shape[1], tn, 1)
                v = np.concatenate((v1,v2),axis=2)
            v = np.reshape(v, newshape=[1, shape[1], tn, 1, 1, 2])
        new_state_dict[k] = torch.from_numpy(v)
    return new_state_dict


# model registration
def register_model(fn):
    mod = sys.modules[fn.__module__]
    model_name = fn.__name__

    # add entries to registry dict/sets
    assert model_name not in model_dict
    model_dict[model_name] = fn
    if hasattr(mod, 'transfer_weights'):
        transfer_dict[model_name] = mod.transfer_weights
    else:
        transfer_dict[model_name] = None
    return fn
