import os
import sys
import pytest, time
import torch, math
import triton, pickle
import triton.language as tl
import torch.nn.functional as F
import warnings
#import sageattention 
import numpy as np
warnings.filterwarnings("ignore")

from sageattention import sageattn
#attn_output = sageattn(q, k, v, tensor_layout="HND", is_causal=False)

def precision_metric(quant_o, fa2_o, round_num=4): 
    if quant_o.shape[-2] > 200000:
        quant_o, fa2_o = quant_o.cpu(), fa2_o.cpu()
    x, xx = quant_o.float(), fa2_o.float() 
    sim = F.cosine_similarity(x.reshape(1, -1), xx.reshape(1, -1)).item()
    l1 =   ( (x - xx).abs().sum() / xx.abs().sum() ).item()
    rmse = torch.sqrt(torch.mean((x -xx) ** 2)).item()
    sim = round(sim, round_num)
    l1 = round(l1, round_num)
    rmse = round(rmse, round_num)
    print(f'Cossim: {sim:.6f}, L1: {l1:.6f}, RMSE:{rmse:.6f}')   # 0.99+, 0.005-, 

is_causal = False


if __name__ == '__main__':
    with open(f"/root/xxm/data/0_q_tensor.pkl", "rb") as f:
        q = pickle.load(f).to('cuda').to(torch.float16).contiguous()
    with open(f"/root/xxm/data/0_k_tensor.pkl", "rb") as f:
        k = pickle.load(f).to('cuda').to(torch.float16).contiguous()
    with open(f"/root/xxm/data/0_v_tensor.pkl", "rb") as f:
        v = pickle.load(f).to('cuda').to(torch.float16).contiguous()

    # q_flat = q.view(-1)
    # k_flat = k.view(-1)
    # v_flat = v.view(-1)
    # print(f'num_q: {len(q_flat)}, num_k: {len(k_flat)}, num_v: {len(v_flat)}')
    # print(f'q shape: {q.shape}, k shape: {k.shape}, v shape: {v.shape}')

    torch_out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
    #a_flat = torch_out.view(-1)
    #for i in range(len(a_flat)):
    #    print(f'torch_out[{i}]: {a_flat[i]}')

    filename = f"o_fp16.bin"
    num_element = torch_out.numel()
    with open(filename, "rb") as f:
        data = np.fromfile(f, dtype=np.float16, count=num_element)

    attn_output = torch.from_numpy(data).to(torch.float16).to('cuda').contiguous()
    attn_output = attn_output.view(torch_out.shape)

    #attn_output =sageattn(q, k, v, is_causal=is_causal)
    #b_flat = attn_output.view(-1)
    #for i in range(len(b_flat)):
    #    print(f'attn_output[{i}]: {b_flat[i]}')

    precision_metric(attn_output, torch_out)