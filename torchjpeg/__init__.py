"""
torchjpeg
====================================
torchjpeg provides an API for accessing low-level JPEG related constructs directly from pytorch.
"""

import torch

import torchjpeg.dct as dct
from torchjpeg.quantization.ijg import mask_coefficients, compress_coefficients, decompress_coefficients

def ste_round(tensor):
    return tensor + tensor.round().detach() - tensor.detach()

def differentiable_jpeg_compression(image, qualities):
    H, W = image.shape[-2:]
    ycbcr = dct.to_ycbcr(dct.pad_to_block_multiple(image), data_range=1.0)
    y, cb, cr = ycbcr.split(1, dim=1)
    
    y_dct = compress_coefficients(y, qualities, 'luma', ste_round)
    cb_dct = compress_coefficients(cb, qualities, 'chroma', ste_round)
    cr_dct = compress_coefficients(cr, qualities, 'chroma', ste_round)
    
    cb_dct = dct.double_nn_dct(dct.half_nn_dct(cb_dct))
    cr_dct = dct.double_nn_dct(dct.half_nn_dct(cr_dct))
    
    y = decompress_coefficients(y_dct, qualities, 'luma')
    cb = decompress_coefficients(cb_dct, qualities, 'chroma')
    cr = decompress_coefficients(cr_dct, qualities, 'chroma')
    
    return dct.to_rgb(torch.cat((y, cb, cr), dim=1), data_range=1.0).clamp(0, 1)[..., :H, :W]

def jpeg_mask(image, qualities):
    H, W = image.shape[-2:]
    ycbcr = dct.to_ycbcr(dct.pad_to_block_multiple(image), data_range=1.0)
    y, cb, cr = ycbcr.split(1, dim=1)
    
    y = mask_coefficients(y, qualities, 'luma')
    cb = mask_coefficients(cb, qualities, 'chroma', downsample=True)
    cr = mask_coefficients(cr, qualities, 'chroma', downsample=True)
    
    return dct.to_rgb(torch.cat((y, cb, cr), dim=1), data_range=1.0).clamp(0, 1)[..., :H, :W]
