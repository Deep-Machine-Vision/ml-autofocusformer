# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

import torch
from torch import nn
from clusten import CLUSTENQKFunction
import time
import statistics
import gc

"""
Test the correctness and profile the performance of QK custom kernel
With separate forward and backward timings.
"""

def profile_memory():
    """Get current GPU memory usage"""
    if torch.cuda.is_available():
        return {
            'allocated': torch.cuda.memory_allocated() / 1024**2,  # MB
            'reserved': torch.cuda.memory_reserved() / 1024**2,    # MB
            'max_allocated': torch.cuda.max_memory_allocated() / 1024**2  # MB
        }
    return {'allocated': 0, 'reserved': 0, 'max_allocated': 0}


def test_correctness():
    """Test correctness of the custom kernel"""
    print("="*60)
    print("CORRECTNESS TESTING")
    print("="*60)
    
    b, h, n, m, c = 256, 4, 196, 48, 32

    # NOTE: CLUSTENQKFunction transposes key from (b,h,n,c) to (b,h,c,n) before calling CUDA kernel
    # So our reference implementation needs to account for this
    
    query = nn.Parameter(torch.randn(b, h, n, c)).cuda()
    query.retain_grad()
    key = nn.Parameter(torch.randn(b, h, n, c)).cuda()
    key.retain_grad()
    nn_idx = torch.randint(n, (b, n, m)).cuda()

    # Custom kernel (key gets transposed inside CLUSTENQKFunction)
    attn = CLUSTENQKFunction.apply(query, key, nn_idx)
    attn.mean().backward()
    grad_query = query.grad.clone().detach(); query.grad.data.zero_()
    grad_key = key.grad.clone().detach(); key.grad.data.zero_()

    # PyTorch equivalent (matches the reference implementation)
    key_gather = key.gather(
        index=nn_idx.reshape(b, 1, -1, 1).expand(-1, h, -1, c),
        dim=2
    ).reshape(b, h, n, m, c)
    attn2 = (query.unsqueeze(3) * key_gather).sum(-1)
    attn2.mean().backward()
    grad_query2 = query.grad.clone().detach(); query.grad.data.zero_()
    grad_key2 = key.grad.clone().detach(); key.grad.data.zero_()

    forward_diff = torch.linalg.norm(attn2 - attn).item()
    grad_query_diff = torch.linalg.norm(grad_query2 - grad_query).item()
    grad_key_diff = torch.linalg.norm(grad_key2 - grad_key).item()
    
    print(f"Forward pass difference: {forward_diff:.2e}")
    print(f"Query gradient difference: {grad_query_diff:.2e}")
    print(f"Key gradient difference: {grad_key_diff:.2e}")
    
    tol = 3e-3
    passed = forward_diff < tol and grad_query_diff < tol and grad_key_diff < tol
    print(f"\nCorrectness test: {'PASSED' if passed else 'FAILED'}")
    return passed


def profile_kernel(b=256, h=4, n=196, m=48, c=32, num_warmup=10, num_trials=50):
    """Profile custom kernel and PyTorch equivalent with separate fwd/bwd timings"""
    print("\n" + "="*60)
    print(f"PERFORMANCE PROFILING (b={b}, h={h}, n={n}, m={m}, c={c})")
    print(f"Warmup: {num_warmup}, Trials: {num_trials}")
    print("="*60)

    query = nn.Parameter(torch.randn(b, h, n, c)).cuda()
    key = nn.Parameter(torch.randn(b, h, n, c)).cuda()
    nn_idx = torch.randint(n, (b, n, m)).cuda()

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    def time_custom():
        # Forward
        fwd_start = torch.cuda.Event(enable_timing=True)
        fwd_end = torch.cuda.Event(enable_timing=True)
        fwd_start.record()
        out = CLUSTENQKFunction.apply(query, key, nn_idx)
        fwd_end.record()
        torch.cuda.synchronize()
        fwd_time = fwd_start.elapsed_time(fwd_end)

        # Backward
        bwd_start = torch.cuda.Event(enable_timing=True)
        bwd_end = torch.cuda.Event(enable_timing=True)
        bwd_start.record()
        out.mean().backward()
        bwd_end.record()
        torch.cuda.synchronize()
        bwd_time = bwd_start.elapsed_time(bwd_end)

        # zero grads
        if query.grad is not None: query.grad.data.zero_()
        if key.grad is not None: key.grad.data.zero_()
        return fwd_time, bwd_time, torch.cuda.max_memory_allocated() / 1024**2

    def time_pytorch():
        fwd_start = torch.cuda.Event(enable_timing=True)
        fwd_end = torch.cuda.Event(enable_timing=True)
        fwd_start.record()
        kg = key.gather(
            index=nn_idx.reshape(b, 1, -1, 1).expand(-1, h, -1, c),
            dim=2
        ).reshape(b, h, n, m, c)
        out = (query.unsqueeze(3) * kg).sum(-1)
        fwd_end.record()
        torch.cuda.synchronize()
        fwd_time = fwd_start.elapsed_time(fwd_end)

        bwd_start = torch.cuda.Event(enable_timing=True)
        bwd_end = torch.cuda.Event(enable_timing=True)
        bwd_start.record()
        out.mean().backward()
        bwd_end.record()
        torch.cuda.synchronize()
        bwd_time = bwd_start.elapsed_time(bwd_end)

        if query.grad is not None: query.grad.data.zero_()
        if key.grad is not None: key.grad.data.zero_()
        return fwd_time, bwd_time, torch.cuda.max_memory_allocated() / 1024**2

    # Warmup
    for _ in range(num_warmup): time_custom(); time_pytorch()
    torch.cuda.synchronize()

    custom_fwd, custom_bwd, custom_mem = [], [], []
    for _ in range(num_trials):
        fwd_t, bwd_t, mem_peak = time_custom()
        custom_fwd.append(fwd_t); custom_bwd.append(bwd_t); custom_mem.append(mem_peak)

    pytorch_fwd, pytorch_bwd, pytorch_mem = [], [], []
    for _ in range(num_trials):
        fwd_t, bwd_t, mem_peak = time_pytorch()
        pytorch_fwd.append(fwd_t); pytorch_bwd.append(bwd_t); pytorch_mem.append(mem_peak)

    def stats(x):
        return {
            'mean': statistics.mean(x),
            'median': statistics.median(x),
            'std': statistics.stdev(x) if len(x)>1 else 0,
            'min': min(x),
            'max': max(x)
        }

    c_fwd_stats = stats(custom_fwd)
    c_bwd_stats = stats(custom_bwd)
    p_fwd_stats = stats(pytorch_fwd)
    p_bwd_stats = stats(pytorch_bwd)
    c_mem_stats = stats(custom_mem)
    p_mem_stats = stats(pytorch_mem)

    print("\nSeparate Forward/Backward Timing:")
    print(f"{'Pass':<10}{'Custom Mean (ms)':<20}{'Torch Mean (ms)':<20}{'Speedup':<10}")
    print(f"{'FWD':<10}{c_fwd_stats['mean']:<20.3f}{p_fwd_stats['mean']:<20.3f}{p_fwd_stats['mean']/c_fwd_stats['mean']:<10.2f}x")
    print(f"{'BWD':<10}{c_bwd_stats['mean']:<20.3f}{p_bwd_stats['mean']:<20.3f}{p_bwd_stats['mean']/c_bwd_stats['mean']:<10.2f}x")

    print("\nMemory Peak (MB) Mean:")
    print(f"{'Custom':<10}{c_mem_stats['mean']:<10.1f}{'PyTorch':<10}{p_mem_stats['mean']:<10.1f}{'Ratio':<10}{p_mem_stats['mean']/c_mem_stats['mean']:<.2f}")

    return {
        'c_fwd_stats': c_fwd_stats,
        'c_bwd_stats': c_bwd_stats,
        'p_fwd_stats': p_fwd_stats,
        'p_bwd_stats': p_bwd_stats,
        'mem_ratio': p_mem_stats['mean']/c_mem_stats['mean']
    }


def profile_different_sizes():
    """Profile different input sizes with separate fwd/bwd timings"""
    configs = [
        {'b':128,'h':4,'n':196,'m':48,'c':32},
        {'b':256,'h':4,'n':196,'m':48,'c':32},
        {'b':512,'h':8,'n':196,'m':48,'c':64},
        {'b':256,'h':4,'n':392,'m':96,'c':32},
        {'b':4, 'h':4, 'n':128*128, 'm':9, 'c':32},
        {'b':2, 'h':4, 'n':256*256, 'm':9, 'c':32},
        {'b':1, 'h':4, 'n':512*512, 'm':9, 'c':32},
    ]
    results = []
    for cfg in configs:
        print(f"\nTesting config: {cfg}")
        try:
            res = profile_kernel(num_warmup=5, num_trials=20, **cfg)
            res['config'] = cfg
            results.append(res)
        except RuntimeError as e:
            print(f"Config failed: {e}")

    print(f"\n{'Config':<30}{'C_FWD(ms)':<12}{'T_FWD(ms)':<12}{'C_BWD(ms)':<12}{'T_BWD(ms)':<12}{'Mem_Ratio':<10}")
    for r in results:
        c=r['config'];
        name=f"b{c['b']}_h{c['h']}_n{c['n']}_m{c['m']}_c{c['c']}"
        print(f"{name:<30}{r['c_fwd_stats']['mean']:<12.2f}{r['p_fwd_stats']['mean']:<12.2f}"
              f"{r['c_bwd_stats']['mean']:<12.2f}{r['p_bwd_stats']['mean']:<12.2f}{r['mem_ratio']:<10.2f}")

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA is not available. This test requires GPU.")
        exit(1)
    print("CLUSTENQK Kernel Testing and Profiling (Separate Timing)")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name()}")
    if test_correctness():
        profile_kernel()
        profile_different_sizes()
    else:
        print("Correctness test failed! Skipping performance profiling.")
        print("Please check the custom kernel implementation.")