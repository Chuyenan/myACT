import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from gact.conf import config
import gact.cpp_extension.quantization as ext_quantization
import gact.cpp_extension.minimax as ext_minimax
from torch.utils.checkpoint import checkpoint

num_hash_functions = 10
hash_function_length = 16

static_hash_functions = np.random.randint(0, 2, (num_hash_functions, hash_function_length), dtype=np.int32)

def compute_group_hash(group, hash_functions):
    flat_group = group.flatten().cpu().numpy()
    hash_values = np.dot(flat_group, hash_functions.T) > 0
    hash_value = ''.join(map(str, hash_values.astype(int)))
    return hash_value

@torch.no_grad()
def no_scheme_quantize_pack(input, q_bit, seed):
    N = (input.numel() + config.group_size - 1) //  config.group_size
    num_ele = N * config.group_size
    pad_num = num_ele - input.numel() 
    if pad_num > 0:
        input = torch.cat([input.reshape(1, -1), torch.zeros([1, pad_num], 
                                            dtype=input.dtype, device=input.device)], dim=1)

    input_groups = input.reshape(-1, config.group_size)  
          
    if q_bit == 32:  # TODO, use kernel to optimize this
        q_min = input_groups.min(dim=-1, keepdim=True).values
        q_scale = input_groups.max(dim=-1, keepdim=True).values - q_min
        q_input = input_groups
    else:
        q_min, mx = ext_minimax.minimax(input_groups)
        input_groups = input_groups.view(N, -1, config.group_size)
        q_min = q_min.reshape(N, -1, 1)
        mx = mx.view(N, -1, 1)
        q_input, q_scale = ext_quantization.pack_single_precision(input_groups, q_min, mx, q_bit, True, seed)
        del mx
        
    # q_input_numel = q_input.numel()
    # input_drop_idx = q_input.abs().topk(int(q_input_numel * 0.9), sorted=False)[1]
    # input_drop_idx = input_drop_idx.to(torch.int32)
    # q_input = q_input[input_drop_idx]
        
    return q_input, q_scale, q_min
    # return q_input, q_scale, q_min, q_input_numel, input_drop_idx

@torch.no_grad()
def dequantize_and_unpack(data, shape, q_bit, scale, mn):
    if not isinstance(q_bit, int):
        print("bits must be intergers, now bits ", q_bit)
        assert(False)
        
    if q_bit == 32:
        return data
        
    group_size = config.group_size
    unpack_func = ext_quantization.unpack_single_precision
    num_groups = (int(np.prod(shape)) + group_size - 1)  // group_size
    return unpack_func(
        data, q_bit, scale, mn, num_groups, 1, group_size
    )

def op_quantize_mask(input):
    return [ext_quantization.act_quantize_dropout_mask(input.contiguous()), input.shape]

def op_dequantize_mask(input):
    q_mask, input_shape = input
    output = ext_quantization.act_dequantize_dropout_mask(q_mask, np.prod(input_shape)).reshape(input_shape)
    return output

@torch.no_grad()
def no_scheme_compute_quantization_bits(input, group_size):
    N = input.shape[0]
    D = input.shape[1]
    input_flatten = input.view(N, -1)
    num_features = input_flatten.shape[1]
    num_pixels = num_features // D

    # Compute min, max by groups
    if num_features % group_size != 0:
        # Padding
        new_num_features = (num_features // group_size + 1) * group_size
        delta = new_num_features - num_features
        input_flatten = torch.cat([input_flatten,
                                   torch.zeros([N, delta], dtype=input.dtype, device=input.device)], 1)

    input_groups = input_flatten.view(-1, group_size)
    mn, mx = ext_minimax.minimax(input_groups)

    b = 1
    return input_groups.view(N, -1, group_size), b, mn.view(N, -1, 1), mx.view(N, -1, 1)

def cvbn_scheme_quantize_pack(input, q_bit, seed):
    B, C, H, W = input.shape

    if H == 1:
        return input, None, None, None, None, None, None, None, None, None, None, None

    pool_kernel_size = 8 if H >= 8 else H
    lfc_input = F.avg_pool2d(input, pool_kernel_size, stride=pool_kernel_size, padding=0)
    input_lfc_large = F.upsample_nearest(lfc_input, size=(H, W), scale_factor=None)
    hfc_input = input - input_lfc_large
    lfc_shape = lfc_input.shape
    hfc_shape = hfc_input.shape
    
    featuremap_area_lfc = lfc_input.shape[-2:].numel()  # should be n
    if featuremap_area_lfc > config.group_size:
        group_size = config.group_size
        input_lfc_groups = lfc_input.reshape(B, -1, group_size)
        q_bits_l = 8
        q_min_l = input_lfc_groups.min(dim=-1).values.unsqueeze(dim=-1)
        mx_l = input_lfc_groups.max(dim=-1).values.unsqueeze(dim=-1)
    else:
        group_size = featuremap_area_lfc
        input_lfc_groups = lfc_input.reshape(B, -1, group_size)
        q_bits_l = 8
        q_min_l = input_lfc_groups.min(dim=-1).values.unsqueeze(dim=-1)
        mx_l = input_lfc_groups.max(dim=-1).values.unsqueeze(dim=-1)
    
    featuremap_area = hfc_input.shape[-2:].numel()  # should be n
    if featuremap_area > config.group_size:
        group_size = config.group_size
        input_hfc_groups = hfc_input.reshape(B, -1, group_size)
        q_bits = 2
        q_min = input_hfc_groups.min(dim=-1).values.unsqueeze(dim=-1)
        mx = input_hfc_groups.max(dim=-1).values.unsqueeze(dim=-1)
    else:
        group_size = featuremap_area
        input_hfc_groups = hfc_input.reshape(B, -1, group_size)
        q_bits = 2
        q_min = input_hfc_groups.min(dim=-1).values.unsqueeze(dim=-1)
        mx = input_hfc_groups.max(dim=-1).values.unsqueeze(dim=-1)

    quantized_results = {}

    lfc_packed = []
    for group in input_lfc_groups:
        hash_value = compute_group_hash(group, static_hash_functions)
        if hash_value in quantized_results:
            lfc_packed.append(quantized_results[hash_value])
        else:
            packed, q_scale_l = ext_quantization.pack_single_precision(group.unsqueeze(0), q_min_l, mx_l, q_bits_l, True, seed)
            quantized_results[hash_value] = packed
            lfc_packed.append(packed)
    lfc_packed = torch.cat(lfc_packed, dim=0)

    hfc_packed = []
    for group in input_hfc_groups:
        hash_value = compute_group_hash(group, static_hash_functions)
        if hash_value in quantized_results:
            hfc_packed.append(quantized_results[hash_value])
        else:
            packed, q_scale = ext_quantization.pack_single_precision(group.unsqueeze(0), q_min, mx, q_bits, True, seed)
            quantized_results[hash_value] = packed
            hfc_packed.append(packed)
    hfc_packed = torch.cat(hfc_packed, dim=0)
    
    lfc_numel = lfc_packed.numel()
    hfc_numel = hfc_packed.numel()
    
    if lfc_numel > 0:
        lfc_topk_k = max(1, int(lfc_numel * 0.9))
        lfc_flat = lfc_packed.flatten()
        lfc_drop_idx_flat = lfc_flat.abs().topk(lfc_topk_k, sorted=False)[1]
        lfc_packed = lfc_flat[lfc_drop_idx_flat].view(-1, lfc_topk_k)
        lfc_drop_idx = lfc_drop_idx_flat.to(torch.int32)
    else:
        lfc_packed = torch.tensor([], dtype=lfc_packed.dtype, device=lfc_packed.device)
        lfc_drop_idx = torch.tensor([], dtype=torch.int32, device=lfc_packed.device)
    
    if hfc_numel > 0:
        hfc_topk_k = max(1, int(hfc_numel * 0.1))
        hfc_flat = hfc_packed.flatten()
        hfc_drop_idx_flat = hfc_flat.abs().topk(hfc_topk_k, sorted=False)[1]
        hfc_packed = hfc_flat[hfc_drop_idx_flat].view(-1, hfc_topk_k)
        hfc_drop_idx = hfc_drop_idx_flat.to(torch.int32)
    else:
        hfc_packed = torch.tensor([], dtype=hfc_packed.dtype, device=hfc_packed.device)
        hfc_drop_idx = torch.tensor([], dtype=torch.int32, device=hfc_packed.device)
    
    return lfc_packed, lfc_shape, lfc_numel, q_scale_l, q_min_l, lfc_drop_idx, hfc_packed, hfc_shape, hfc_numel, q_scale, q_min, hfc_drop_idx

def op_quantize_cvbn(input, q_bit, seed):
    # input_lfc, hfc_input, q_scale, q_min = cvbn_scheme_quantize_pack(input, q_bit, seed)
    lfc_input, lfc_shape, lfc_numel, q_scale_l, q_min_l, lfc_drop_idx, hfc_input, hfc_shape, hfc_numel, q_scale, q_min, hfc_drop_idx = cvbn_scheme_quantize_pack(input, q_bit, seed)
    return [lfc_input, lfc_shape, lfc_numel, q_scale_l, q_min_l, lfc_drop_idx, 8, hfc_input, hfc_shape, hfc_numel, q_scale, q_min, hfc_drop_idx, 2]
    # return [input_lfc, hfc_input, 2, q_scale, q_min]

@torch.no_grad()
def cvbn_dequantize_and_unpack(data, shape, bits, scale, mn, group_size):

    # Pad to group_size
    N = shape[0]
    num_features = int(shape[1:].numel())

    num_features = (num_features + (group_size - num_features % group_size) % group_size)

    # Unpack bitstream
    if isinstance(bits, int):
        unpack_func = ext_quantization.unpack_single_precision

    data = unpack_func(data, bits, scale, mn, N, num_features // group_size, group_size)
    
    unpack_func(
        data, q_bit, scale, mn, num_groups, 1, group_size
    )

    return data
    
@torch.no_grad()
def op_dequantize_cvbn(input, input_shape):
    B, C, H, W = input_shape
    if H == 1:
        x, _, _, _, _,_,_,_,_,_,_,_,_,_ = input
        return x
    lfc_input, lfc_shape, lfc_numel, q_scale_l, q_min_l, lfc_drop_idx, q_bit_l, hfc_input, hfc_shape, hfc_numel, q_scale, q_min, hfc_drop_idx, q_bit = input
    # lfc_input, hfc_input, q_bit, q_scale, q_min = input
    
    
    featuremap_area_l = lfc_shape[-2:].numel()
    group_size = config.group_size if featuremap_area_l > config.group_size else featuremap_area_l
    # group_size = 256
    num_groups = (int(np.prod(lfc_shape)) + group_size - 1)  // group_size
    
    lfc_input = torch.zeros(lfc_numel, device=lfc_input.device, dtype=lfc_input.dtype).scatter_(0, lfc_drop_idx.to(torch.int64), lfc_input)
    hfc_input = torch.zeros(hfc_numel, device=hfc_input.device, dtype=hfc_input.dtype).scatter_(0, hfc_drop_idx.to(torch.int64), hfc_input)
    
    x_lfc_dequant = ext_quantization.unpack_single_precision(
        lfc_input, q_bit_l, q_scale_l, q_min_l, num_groups, 1, group_size
    )
    num_features = lfc_shape[1:].numel()
    lfc_input = x_lfc_dequant.view(lfc_shape[0], -1)[:, :num_features]
    lfc_input = x_lfc_dequant.view(*lfc_shape).contiguous()
    
    featuremap_area = input_shape[-2:].numel()
    group_size = config.group_size if featuremap_area > config.group_size else featuremap_area
    num_groups = (int(np.prod(input_shape)) + group_size - 1)  // group_size
    x_hfc_dequant = ext_quantization.unpack_single_precision(
        hfc_input, q_bit, q_scale, q_min, num_groups, 1, group_size
    )
    num_features = input_shape[1:].numel()
    hfc_input = x_hfc_dequant.view(input_shape[0], -1)[:, :num_features]
    hfc_input = x_hfc_dequant.view(*input_shape).contiguous()
    
    lfc_input = F.upsample_nearest(lfc_input.to(torch.float32), size=(H, W), scale_factor=None)
    
    # lfc_input = lfc_input.to(torch.float32)
    return lfc_input + hfc_input
   
@torch.no_grad()
def op_quantize(input, q_bit, seed):
    q_input, q_scale, q_min = no_scheme_quantize_pack(input, q_bit, seed)
    # q_input, q_scale, q_min, q_input_numel, input_drop_idx = no_scheme_quantize_pack(input, q_bit, seed)
    # return [q_input, q_bit, q_scale, q_min, q_input_numel, input_drop_idx]
    return [q_input, q_bit, q_scale, q_min]

@torch.no_grad()
def op_dequantize(input, input_shape):
    # q_input, q_bit, q_scale, q_min, q_input_numel, input_drop_idx = input
    q_input, q_bit, q_scale, q_min = input
    # q_input = torch.zeros(q_input_numel, device=q_input.device, dtype=q_input.dtype).scatter_(0, input_drop_idx.to(torch.int64), q_input)
    input = dequantize_and_unpack(
        q_input, input_shape, q_bit, q_scale, q_min)

    num_features = np.prod(input_shape)
    input = input.ravel()[:num_features]
    input = input.reshape(*input_shape).contiguous()
    return input

# Implementation of efficient self attention
# https://arxiv.org/abs/2112.05682
def self_atten(dropout_p, query_layer, key_layer, value_layer,
               q_chunk_size, k_chunk_size, use_checkpoint=True):
    batch_size, num_heads, seq_len, q_features = query_layer.shape
    batch_size, num_heads, seq_len, k_features = key_layer.shape
    batch_size, num_heads, seq_len, v_features = value_layer.shape
    q_chunk_size = min(q_chunk_size, seq_len)
    dropout = nn.Dropout(dropout_p)

    def _query_chunk_attention(query, key, value):
        batch_size, num_heads, num_kv, k_features = key.shape
        v_features = value.shape[-1]
        key_chunk_size = min(k_chunk_size, num_kv)
        num_key_chunk = math.ceil(num_kv / key_chunk_size)
        query = query / math.sqrt(k_features)

        def summarize_chunk(query, key, value):
            attn_weights = torch.einsum('bhqd,bhkd->bhqk', query, key)
            max_score = torch.max(attn_weights, axis=-1, keepdims=True).values
            max_score = max_score.detach()
            exp_weights = torch.exp(attn_weights - max_score)
            exp_values = torch.einsum('bhvf,bhqv->bhqf', value, exp_weights)
            exp_values = dropout(exp_values)
            return (exp_values, exp_weights.sum(axis=-1), max_score.squeeze())

        chunk_values = None
        chunk_weights = None
        global_max = None

        def batch_dot(m1, m2):
            feature_size = m1.shape[-1]
            v = m1.reshape(-1, feature_size) * m2.reshape(-1, 1)
            return v.reshape(m1.shape)

        for i in range(num_key_chunk):
            key_chunk = key[:, :, i *
                            key_chunk_size: (i+1) * key_chunk_size, :]
            value_chunk = value[:, :, i *
                                key_chunk_size: (i+1) * key_chunk_size, :]
            if use_checkpoint:
                chunk_value, chunk_weight, chunk_max = \
                    checkpoint(
                        summarize_chunk, query, key_chunk, value_chunk)
            else:
                chunk_value, chunk_weight, chunk_max = summarize_chunk(
                    query, key_chunk, value_chunk)

            if global_max is None:
                global_max = chunk_max
                chunk_values = chunk_value
                chunk_weights = chunk_weight
            else:
                old_max = global_max
                global_max = torch.maximum(chunk_max, global_max).detach()

                diff1 = torch.exp(chunk_max - global_max).detach()
                chunk_value = batch_dot(chunk_value, diff1)
                chunk_weight *= diff1

                diff2 = torch.exp(old_max - global_max).detach()
                chunk_values = batch_dot(chunk_values, diff2)
                chunk_weights *= diff2

                chunk_values += chunk_value
                chunk_weights += chunk_weight

        chunk_values = chunk_values.reshape(-1, chunk_values.shape[-1])
        chunk_weights = chunk_weights.reshape(-1, 1)
        return chunk_values / chunk_weights

    num_q_chunk = math.ceil(query_layer.shape[2] / q_chunk_size)
    res = torch.zeros(query_layer.shape).cuda()
    for i in range(num_q_chunk):
        r = _query_chunk_attention(query_layer[:, :, i*q_chunk_size:(i+1)*q_chunk_size, :],
                                   key_layer, value_layer)
        res[:, :, i*q_chunk_size:(i+1)*q_chunk_size, :] = r.reshape(
            batch_size, num_heads, q_chunk_size, q_features)
    return res
