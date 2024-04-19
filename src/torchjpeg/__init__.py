"""
torchjpeg
====================================
torchjpeg provides an API for accessing low-level JPEG related constructs directly from pytorch.
"""

import torch

import torchjpeg.dct as dct
from torchjpeg.quantization.ijg import compress_coefficients, decompress_coefficients

def ste_round(tensor):
    return tensor + tensor.round().detach() - tensor.detach()

def differentiable_jpeg_compression(image, qf):
    H, W = image.shape[-2:]
    ycbcr = dct.to_ycbcr(dct.pad_to_block_multiple(image), data_range=1.0)
    y, cb, cr = ycbcr.split(1, dim=1)
    
    y_dct = compress_coefficients(y, qf, 'luma', ste_round)
    cb_dct = compress_coefficients(cb, qf, 'chroma', ste_round)
    cr_dct = compress_coefficients(cr, qf, 'chroma', ste_round)
    
    cb_dct = dct.double_nn_dct(dct.half_nn_dct(cb_dct))
    cr_dct = dct.double_nn_dct(dct.half_nn_dct(cr_dct))
    
    y = decompress_coefficients(y_dct, qf, 'luma')
    cb = decompress_coefficients(cb_dct, qf, 'chroma')
    cr = decompress_coefficients(cr_dct, qf, 'chroma')
    
    return dct.to_rgb(torch.cat((y, cb, cr), dim=1), data_range=1.0).clamp(0, 1)[..., :H, :W]
