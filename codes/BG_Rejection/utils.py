import math
import numpy as np
import torch
from torch.nn import init


def crop_patches(img, patch_size, stride_size):
    T, H, W = img.shape
    patch_num = (math.ceil((T - patch_size[0]) / stride_size[0]) + 1,
                 math.ceil((H - patch_size[1]) / stride_size[1]) + 1,
                 math.ceil((W - patch_size[2]) / stride_size[2]) + 1)

    patches = []
    for i in range(patch_num[0]):
        for j in range(patch_num[1]):
            for k in range(patch_num[2]):
                t_start = i * stride_size[0] if i < patch_num[0] - 1 else T - patch_size[0]
                h_start = j * stride_size[1] if j < patch_num[1] - 1 else H - patch_size[1]
                w_start = k * stride_size[2] if k < patch_num[2] - 1 else W - patch_size[2]
                patches.append(img[t_start:t_start + patch_size[0],
                               h_start:h_start + patch_size[1],
                               w_start:w_start + patch_size[2]])
    return patches, patch_num


def concat_patches(patches, origin_shape, patch_num, patch_size, stride_size, overlap_size):
    assert overlap_size[0] % 2 == 0 and overlap_size[1] % 2 == 0 and overlap_size[2] % 2 == 0

    out = np.zeros(origin_shape, dtype=np.float32)

    for i in range(patch_num[0]):
        for j in range(patch_num[1]):
            for k in range(patch_num[2]):
                t_start = 0 if i == 0 else i * stride_size[0] + overlap_size[0] // 2
                h_start = 0 if j == 0 else j * stride_size[1] + overlap_size[1] // 2
                w_start = 0 if k == 0 else k * stride_size[2] + overlap_size[2] // 2

                t_end = (i + 1) * stride_size[0] + overlap_size[0] // 2 if i < patch_num[0] - 1 else origin_shape[0]
                h_end = (j + 1) * stride_size[1] + overlap_size[1] // 2 if j < patch_num[1] - 1 else origin_shape[1]
                w_end = (k + 1) * stride_size[2] + overlap_size[2] // 2 if k < patch_num[2] - 1 else origin_shape[2]

                if i == 0:
                    pt_start = 0
                    pt_end = patch_size[0] - (overlap_size[0] // 2)
                elif i == patch_num[0] - 1:
                    pt_start = patch_size[0] - (t_end - t_start)
                    pt_end = patch_size[0]
                else:
                    pt_start = overlap_size[0] // 2
                    pt_end = patch_size[0] - (overlap_size[0] // 2)

                if j == 0:
                    ph_start = 0
                    ph_end = patch_size[1] - (overlap_size[1] // 2)
                elif j == patch_num[1] - 1:
                    ph_start = patch_size[1] - (h_end - h_start)
                    ph_end = patch_size[1]
                else:
                    ph_start = overlap_size[1] // 2
                    ph_end = patch_size[1] - (overlap_size[1] // 2)

                if k == 0:
                    pw_start = 0
                    pw_end = patch_size[2] - (overlap_size[2] // 2)
                elif k == patch_num[2] - 1:
                    pw_start = patch_size[2] - (w_end - w_start)
                    pw_end = patch_size[2]
                else:
                    pw_start = overlap_size[2] // 2
                    pw_end = patch_size[2] - (overlap_size[2] // 2)

                out[t_start:t_end, h_start:h_end, w_start:w_end] = patches[
                                                                       i * patch_num[1] * patch_num[2] + j * patch_num[
                                                                           2] + k] \
                    [pt_start:pt_end, ph_start:ph_end, pw_start:pw_end]
    return out


def collate_fn(batch):
    batch = list(zip(*batch))
    return torch.stack(batch[0]), torch.stack(batch[1])


def init_weights(net, init_type='normal', init_gain=0.02):
    '''
    Initialize network weights.
    Parameters:
        net (network) -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float) -- scaling factor for normal, xavier and orthogonal.
    '''

    # define the initialization function
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm3d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)

    # apply the initialization function
    net.apply(init_func)


def set_requires_grad(nets, requires_grad=False):
    """
    Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad
