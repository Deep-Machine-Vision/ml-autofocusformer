/*
 * For licensing see accompanying LICENSE file.
 * Copyright (C) 2023 Apple Inc. All Rights Reserved.
 */

#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <type_traits>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/ATen.h>
#include <ATen/native/cuda/KernelUtils.cuh>
#include <ATen/AccumulateType.h>

#define CUDA_NUM_THREADS 1024
#define WARP_SIZE 32
#define MAX_SHARED_MEM 48 * 1024  // 48KB shared memory per block

template<typename T>
__device__ __forceinline__ T warp_shfl_down(T val, int offset) {
    return __shfl_down_sync(0xffffffff, val, offset, 32);
}

template<>
__device__ __forceinline__ c10::Half warp_shfl_down<c10::Half>(c10::Half val, int offset) {
    return __shfl_down_sync(0xffffffff, static_cast<__half>(val), offset, 32);
}

template<typename T1, typename T2>
__host__ __device__ constexpr auto cuda_min(T1 a, T2 b) -> typename std::common_type<T1, T2>::type {
    return (a < b) ? static_cast<typename std::common_type<T1, T2>::type>(a) : static_cast<typename std::common_type<T1, T2>::type>(b);
}
template <typename scalar_t>
__global__ void __launch_bounds__(256, 2) clusten_qk_cuda_forward_kernel_opt(
    const torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> query,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> key,
    const torch::PackedTensorAccessor32<int64_t,3,torch::DefaultPtrTraits> nbhd_idx,
    torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> attn,
    const int length, const int batch_size, const int heads, const int nbhd_size, const int dim) {

    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int grid_size = gridDim.x * blockDim.x;
    const int64_t total_work = (int64_t)batch_size * heads * length * nbhd_size;

    for (int64_t work_idx = thread_id; work_idx < total_work; work_idx += grid_size) {

            // Fast 4D index computation using bit shifts and multiplication
            const int bh = work_idx / (length * nbhd_size);
            const int remainder = work_idx - bh * (length * nbhd_size);
            const int i = remainder / nbhd_size;
            const int ni = remainder - i * nbhd_size;

            const int b = bh / heads;
            const int h = bh - b * heads;

            const int64_t nbi = nbhd_idx[b][i][ni];

            int64_t next_nbi = 0;
            const int64_t next_work = work_idx + grid_size;
            if (next_work < total_work) {
                    const int next_bh = next_work / (length * nbhd_size);
                    const int next_remainder = next_work - next_bh * (length * nbhd_size);
                    const int next_i = next_remainder / nbhd_size;
                    const int next_ni = next_remainder - next_i * nbhd_size;
                    const int next_b = next_bh / heads;

                    // Load next neighbor index early to hide memory latency
                    if (next_b < batch_size && next_i < length && next_ni < nbhd_size) {
                            next_nbi = nbhd_idx[next_b][next_i][next_ni];
                    }
            }

            scalar_t result = 0;

            if (dim == 32) {
                    #pragma unroll
                    for (int c = 0; c < 32; ++c) {
                            result += query[b][h][i][c] * key[b][h][c][nbi];
                    }
            } else if (dim == 64) {
                    #pragma unroll
                    for (int c = 0; c < 64; c += 8) {
                            scalar_t sum8 = 0;
                            sum8 += query[b][h][i][c] * key[b][h][c][nbi];
                            sum8 += query[b][h][i][c+1] * key[b][h][c+1][nbi];
                            sum8 += query[b][h][i][c+2] * key[b][h][c+2][nbi];
                            sum8 += query[b][h][i][c+3] * key[b][h][c+3][nbi];
                            sum8 += query[b][h][i][c+4] * key[b][h][c+4][nbi];
                            sum8 += query[b][h][i][c+5] * key[b][h][c+5][nbi];
                            sum8 += query[b][h][i][c+6] * key[b][h][c+6][nbi];
                            sum8 += query[b][h][i][c+7] * key[b][h][c+7][nbi];
                            result += sum8;
                    }
            } else {
                    int c = 0;

                    for (; c + 15 < dim; c += 16) {
                            scalar_t sum16 = 0;
                            sum16 += query[b][h][i][c] * key[b][h][c][nbi];
                            sum16 += query[b][h][i][c+1] * key[b][h][c+1][nbi];
                            sum16 += query[b][h][i][c+2] * key[b][h][c+2][nbi];
                            sum16 += query[b][h][i][c+3] * key[b][h][c+3][nbi];
                            sum16 += query[b][h][i][c+4] * key[b][h][c+4][nbi];
                            sum16 += query[b][h][i][c+5] * key[b][h][c+5][nbi];
                            sum16 += query[b][h][i][c+6] * key[b][h][c+6][nbi];
                            sum16 += query[b][h][i][c+7] * key[b][h][c+7][nbi];
                            sum16 += query[b][h][i][c+8] * key[b][h][c+8][nbi];
                            sum16 += query[b][h][i][c+9] * key[b][h][c+9][nbi];
                            sum16 += query[b][h][i][c+10] * key[b][h][c+10][nbi];
                            sum16 += query[b][h][i][c+11] * key[b][h][c+11][nbi];
                            sum16 += query[b][h][i][c+12] * key[b][h][c+12][nbi];
                            sum16 += query[b][h][i][c+13] * key[b][h][c+13][nbi];
                            sum16 += query[b][h][i][c+14] * key[b][h][c+14][nbi];
                            sum16 += query[b][h][i][c+15] * key[b][h][c+15][nbi];
                            result += sum16;
                    }

                    for (; c + 7 < dim; c += 8) {
                            scalar_t sum8 = 0;
                            sum8 += query[b][h][i][c] * key[b][h][c][nbi];
                            sum8 += query[b][h][i][c+1] * key[b][h][c+1][nbi];
                            sum8 += query[b][h][i][c+2] * key[b][h][c+2][nbi];
                            sum8 += query[b][h][i][c+3] * key[b][h][c+3][nbi];
                            sum8 += query[b][h][i][c+4] * key[b][h][c+4][nbi];
                            sum8 += query[b][h][i][c+5] * key[b][h][c+5][nbi];
                            sum8 += query[b][h][i][c+6] * key[b][h][c+6][nbi];
                            sum8 += query[b][h][i][c+7] * key[b][h][c+7][nbi];
                            result += sum8;
                    }

                    for (; c < dim; ++c) {
                            result += query[b][h][i][c] * key[b][h][c][nbi];
                    }
            }

            attn[b][h][i][ni] = result;
    }
}

template <typename scalar_t>
__global__ void __launch_bounds__(128, 4) clusten_qk_cuda_forward_kernel_warp_opt(
    const torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> query,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> key,
    const torch::PackedTensorAccessor32<int64_t,3,torch::DefaultPtrTraits> nbhd_idx,
    torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> attn,
    const int length, const int batch_size, const int heads, const int nbhd_size, const int dim) {

    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int warps_per_block = blockDim.x / 32;

    // Each warp processes one (batch, head) combo
    const int bh = blockIdx.x * warps_per_block + warp_id;
    if (bh >= batch_size * heads) return;

    const int b = bh / heads;
    const int h = bh % heads;

    const int tokens_per_iteration = 32;

    for (int token_base = blockIdx.y * tokens_per_iteration; token_base < length; token_base += gridDim.y * tokens_per_iteration) {
            const int i = token_base + lane_id;

            if (i < length) {
                    for (int ni = 0; ni < nbhd_size; ++ni) {
                            const int64_t nbi = nbhd_idx[b][i][ni];

                            scalar_t result = 0;

                            for (int c_base = 0; c_base < dim; c_base += 32) {
                                    const int c = c_base + lane_id;
                                    scalar_t local_product = 0;

                                    if (c < dim) {
                                            local_product = query[b][h][i][c] * key[b][h][c][nbi];
                                    }

                                    #pragma unroll
                                    for (int offset = 16; offset > 0; offset /= 2) {
                                            local_product += warp_shfl_down(local_product, offset);
                                    }

                                    if (lane_id == 0) {
                                            result += local_product;
                                    }
                            }

                            if (lane_id == 0) {
                                    attn[b][h][i][ni] = result;
                            }
                    }
            }
    }
}

template <typename scalar_t>
__global__ void clusten_qk_cuda_backward_kernel_opt(
    const torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> d_attn,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> query,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> key,
    const torch::PackedTensorAccessor32<int64_t,3,torch::DefaultPtrTraits> nbhd_idx,
    torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> d_query,
    torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> d_key,
    const int length,
    const int batch_size,
    const int heads,
    const int nbhd_size,
    const int dim,
    const size_t d_key_numel) {

    const int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (z < batch_size * heads) {
            const int i = blockIdx.y * blockDim.y + threadIdx.y;
            if (i < length) {
                    const int c = blockIdx.x * blockDim.x + threadIdx.x;
                    if (c < dim) {
                            const int b = z / heads;
                            const int h = z - b * heads;
                            scalar_t dq_update = 0;

                            #pragma unroll
                            for (int ni = 0; ni < nbhd_size; ++ni) {
                                    const int64_t nbi = nbhd_idx[b][i][ni];
                                    const scalar_t d_attn_val = d_attn[b][h][i][ni];
                                    const scalar_t query_val = query[b][h][i][c];

                                    dq_update += key[b][h][c][nbi] * d_attn_val;

                                    atomicAdd(&d_key[b][h][c][nbi], query_val * d_attn_val);
                            }
                            
                            d_query[b][h][i][c] = dq_update;
                    }
            }
    }
}

// template <typename scalar_t>
// __global__ void clusten_qk_cuda_backward_kernel_opt(
//     const torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> d_attn,
//     const torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> query,
//     const torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> key,
//     const torch::PackedTensorAccessor32<int64_t,3,torch::DefaultPtrTraits> nbhd_idx,
//     torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> d_query,
//     torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> d_key,
//     const int length,
//     const int batch_size,
//     const int heads,
//     const int nbhd_size,
//     const int dim,
//     const size_t d_key_numel) {

//     const int z = blockIdx.z * blockDim.z + threadIdx.z;
//     if (z < batch_size * heads) {
//             const int i = blockIdx.y * blockDim.y + threadIdx.y;
//             if (i < length) {
//                     const int c = blockIdx.x * blockDim.x + threadIdx.x;
//                     if (c < dim) {
//                             const int b = z / heads;
//                             const int h = z - b * heads;
//                             scalar_t dq_update = 0;

//                             #pragma unroll
//                             for (int ni = 0; ni < nbhd_size; ++ni) {
//                                     const int64_t nbi = nbhd_idx[b][i][ni];
//                                     const scalar_t d_attn_val = d_attn[b][h][i][ni];
//                                     const scalar_t query_val = query[b][h][i][c];

//                                     dq_update += key[b][h][c][nbi] * d_attn_val;

//                                     atomicAdd(&d_key[b][h][c][nbi], query_val * d_attn_val);
//                             }

//                             d_query[b][h][i][c] = dq_update;
//                     }
//             }
//     }
// }

template <typename scalar_t>
__global__ void clusten_qk_cuda_backward_kernel_warp_opt(
    const torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> d_attn,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> query,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> key,
    const torch::PackedTensorAccessor32<int64_t,3,torch::DefaultPtrTraits> nbhd_idx,
    torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> d_query,
    torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> d_key,
    const int length,
    const int batch_size,
    const int heads,
    const int nbhd_size,
    const int dim,
    const size_t d_key_numel) {

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warps_per_block = blockDim.x / WARP_SIZE;

    // Each warp processes one (batch, head) combo
    const int bh = blockIdx.x * warps_per_block + warp_id;
    if (bh >= batch_size * heads) return;

    const int b = bh / heads;
    const int h = bh % heads;

    for (int i = blockIdx.y; i < length; i += gridDim.y) {
            for (int c = lane_id; c < dim; c += WARP_SIZE) {
                    scalar_t dq_update = 0;

                    for (int ni = 0; ni < nbhd_size; ++ni) {
                            const int64_t nbi = nbhd_idx[b][i][ni];
                            const scalar_t d_attn_val = d_attn[b][h][i][ni];

                            dq_update += key[b][h][c][nbi] * d_attn_val;

                            atomicAdd(&d_key[b][h][c][nbi], query[b][h][i][c] * d_attn_val);
                    }

                    d_query[b][h][i][c] = dq_update;
            }
    }
}

torch::Tensor clusten_qk_cuda_forward_opt(
    const torch::Tensor &query,
    const torch::Tensor &key,
    const torch::Tensor &nbhd_idx) {

    int64_t batch_size = query.size(0);
    int64_t heads = query.size(1);
    int64_t length = query.size(2);
    int64_t dim = query.size(3);
    int64_t nbhd_size = nbhd_idx.size(2);

    auto attn = torch::zeros(
        {batch_size, heads, length, nbhd_size}, query.options());

    const auto stream = c10::cuda::getCurrentCUDAStream();

    const int64_t total_elements = batch_size * heads * length * nbhd_size;
    const bool use_warp_kernel = (length >= 32768 && dim <= 64 && nbhd_size <= 16);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(query.scalar_type(), "clusten_qk_cuda_forward_opt", ([&] {
        const auto query_a = query.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
        const auto key_a = key.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
        const auto nbhd_idx_a = nbhd_idx.packed_accessor32<int64_t,3,torch::DefaultPtrTraits>();
        auto attn_a = attn.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();

        if (use_warp_kernel) {
                const int warps_per_block = 4;
                const int threads_per_block = warps_per_block * 32;
                const int tokens_per_iteration = 32;

                const dim3 blocks(
                    (batch_size * heads + warps_per_block - 1) / warps_per_block,
                    (length + tokens_per_iteration - 1) / tokens_per_iteration
                );
                const dim3 threads(threads_per_block);

                clusten_qk_cuda_forward_kernel_warp_opt<scalar_t><<<blocks, threads, 0, stream>>>(
                    query_a, key_a, nbhd_idx_a, attn_a, 
                    length, batch_size, heads, nbhd_size, dim);

        } else {
                const int threads_per_block = 256;

                const int target_blocks = 400;
                const int64_t work_per_block = (total_elements + target_blocks - 1) / target_blocks;
                const int64_t actual_blocks = (total_elements + work_per_block - 1) / work_per_block;

                clusten_qk_cuda_forward_kernel_opt<scalar_t><<<actual_blocks, threads_per_block, 0, stream>>>(
                    query_a, key_a, nbhd_idx_a, attn_a, 
                    length, batch_size, heads, nbhd_size, dim);
        }
    }));

    return attn;
}

std::vector<torch::Tensor> clusten_qk_cuda_backward_opt(
    const torch::Tensor &d_attn,
    const torch::Tensor &query,
    const torch::Tensor &key,
    const torch::Tensor &nbhd_idx) {

    int64_t batch_size = query.size(0);
    int64_t heads = query.size(1);
    int64_t length = query.size(2);
    int64_t dim = query.size(3);
    int64_t nbhd_size = nbhd_idx.size(2);

    auto d_query = torch::zeros_like(query);
    auto d_key = torch::zeros_like(key);

    const auto stream = c10::cuda::getCurrentCUDAStream();

    if (length > 32768) {
        const int warps_per_block = 4;
        const int threads_per_block = warps_per_block * WARP_SIZE;

        const dim3 blocks(
                            (batch_size * heads + warps_per_block - 1) / warps_per_block,
                            cuda_min((int64_t)length, 512L)
        );
        const dim3 threads(threads_per_block);

        AT_DISPATCH_FLOATING_TYPES_AND_HALF(query.scalar_type(), "clusten_qk_cuda_backward_warp_opt", ([&] {
                const auto d_attn_a = d_attn.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
                const auto query_a = query.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
                const auto key_a = key.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
                const auto nbhd_idx_a = nbhd_idx.packed_accessor32<int64_t,3,torch::DefaultPtrTraits>();
                auto d_query_a = d_query.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
                auto d_key_a = d_key.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();

                const size_t d_key_numel = d_key.numel();
                clusten_qk_cuda_backward_kernel_warp_opt<scalar_t><<<blocks, threads, 0, stream>>>(
                    d_attn_a, query_a, key_a, nbhd_idx_a, d_query_a, d_key_a,
                    length, batch_size, heads, nbhd_size, dim, d_key_numel);
        }));
    } else {
        int zsize = batch_size * heads;
        int CHANNELTHREADS = cuda_min(int64_t(CUDA_NUM_THREADS), dim);
        int TOKENTHREADS = cuda_min(int64_t(CUDA_NUM_THREADS / CHANNELTHREADS), length);
        int BATCHTHREADS = max(1, CUDA_NUM_THREADS / (TOKENTHREADS * CHANNELTHREADS));

        const dim3 blocks(
                            (dim + CHANNELTHREADS - 1) / CHANNELTHREADS,
                            (length + TOKENTHREADS - 1) / TOKENTHREADS,
                            (zsize + BATCHTHREADS - 1) / BATCHTHREADS);
        const dim3 threads(CHANNELTHREADS, TOKENTHREADS, BATCHTHREADS);

        AT_DISPATCH_FLOATING_TYPES_AND_HALF(query.scalar_type(), "clusten_qk_cuda_backward_opt", ([&] {
                const auto d_attn_a = d_attn.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
                const auto query_a = query.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
                const auto key_a = key.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
                const auto nbhd_idx_a = nbhd_idx.packed_accessor32<int64_t,3,torch::DefaultPtrTraits>();
                auto d_query_a = d_query.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
                auto d_key_a = d_key.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();

                const size_t d_key_numel = d_key.numel();
                clusten_qk_cuda_backward_kernel_opt<scalar_t><<<blocks, threads, 0, stream>>>(
                    d_attn_a, query_a, key_a, nbhd_idx_a, d_query_a, d_key_a,
                    length, batch_size, heads, nbhd_size, dim, d_key_numel);
        }));
    }

    return {d_query, d_key.to(key.dtype())};
}


template <typename scalar_t>
__global__ void clusten_qk_cuda_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> query,      // b x h x n x c
    const torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> key,        // b x h x c x n (reordered by cluster)
    const torch::PackedTensorAccessor32<int64_t,3,torch::DefaultPtrTraits> nbhd_idx,    // b x n x m
    torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> attn,             // b x h x n x m
    const int length,           // n
    const int batch_size,       // b
    const int heads,            // h
    const int nbhd_size,        // m
    const int dim) {            // c

    const int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (z < batch_size * heads){
        const int i = blockIdx.y * blockDim.y + threadIdx.y;
        if (i < length){
            const int ni = blockIdx.x * blockDim.x + threadIdx.x;
            if (ni < nbhd_size){
                const int b = z / heads;
                const int h = z - b * heads;
                int64_t nbi = nbhd_idx[b][i][ni];
                // calculate q@k
                scalar_t updt = 0;
                #pragma unroll
                for (unsigned int c=0; c < dim; ++c) {
                    updt += query[b][h][i][c] * key[b][h][c][nbi];
                }
                attn[b][h][i][ni] = updt;
            }
        }
    }
}

torch::Tensor clusten_qk_cuda_forward(
    const torch::Tensor &query,
    const torch::Tensor &key,
    const torch::Tensor &nbhd_idx) {

    int64_t batch_size = query.size(0);
    int64_t heads = query.size(1);
    int64_t length = query.size(2);
    int64_t dim = query.size(3);
    int64_t nbhd_size = nbhd_idx.size(2);
    int zsize = batch_size * heads;

    int NBHDTHREADS = cuda_min(int64_t(CUDA_NUM_THREADS), nbhd_size);
    int TOKENTHREADS = cuda_min(int64_t(CUDA_NUM_THREADS / NBHDTHREADS), length);
    int BATCHTHREADS = max(1, CUDA_NUM_THREADS / (TOKENTHREADS * NBHDTHREADS));

    auto attn = torch::zeros(
            {batch_size, heads, length, nbhd_size}, query.options());

    const auto stream = c10::cuda::getCurrentCUDAStream();
    const dim3 blocks(
            (nbhd_size + NBHDTHREADS - 1) / NBHDTHREADS,
            (length + TOKENTHREADS - 1) / TOKENTHREADS,
            (zsize + BATCHTHREADS - 1) / BATCHTHREADS);
    const dim3 threads(NBHDTHREADS, TOKENTHREADS, BATCHTHREADS);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(query.scalar_type(), "clusten_qk_cuda_forward", ([&] {
        const auto query_a = query.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
        const auto key_a = key.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
        const auto nbhd_idx_a = nbhd_idx.packed_accessor32<int64_t,3,torch::DefaultPtrTraits>();
        auto attn_a = attn.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();

        clusten_qk_cuda_forward_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                query_a, key_a, nbhd_idx_a, attn_a, 
                length, batch_size, heads, nbhd_size, dim);
    }));
    return attn;
}

template <typename scalar_t>
__global__ void clusten_qk_cuda_backward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> d_attn,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> query,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> key,
    const torch::PackedTensorAccessor32<int64_t,3,torch::DefaultPtrTraits> nbhd_idx,
    torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> d_query,
    torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> d_key,
    const int length,
    const int batch_size,
    const int heads,
    const int nbhd_size,
    const int dim,
    const size_t d_key_numel) {

    const int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (z < batch_size * heads){
        const int i = blockIdx.y * blockDim.y + threadIdx.y;
        if (i < length){
            const int c = blockIdx.x * blockDim.x + threadIdx.x;
            if (c < dim){
                const int b = z / heads;
                const int h = z - b * heads;
                size_t index;
                scalar_t dq_update = 0;
                scalar_t d_attn_tmp;
                #pragma unroll
                for (unsigned int ni=0; ni < nbhd_size; ++ni) {
                    const int64_t nbi = nbhd_idx[b][i][ni];
                    // calculate d_query = key * d_att
                    // calculate d_key = query * d_att
                    d_attn_tmp = d_attn[b][h][i][ni];
                    dq_update += key[b][h][c][nbi] * d_attn_tmp;
                    index = b*d_key.stride(0) + h*d_key.stride(1) + c*d_key.stride(2) + nbi;
                    at::native::fastAtomicAdd(d_key.data(), index, d_key_numel, query[b][h][i][c] * d_attn_tmp, true);
                    //atomicAdd(&(d_key[b][h][c][nbi]), query[b][h][i][c] * d_attn_tmp); // avoid race condition
                }
                d_query[b][h][i][c] = dq_update;
            }
        }
    }
}

std::vector<torch::Tensor> clusten_qk_cuda_backward(
    const torch::Tensor &d_attn,
    const torch::Tensor &query,
    const torch::Tensor &key,
    const torch::Tensor &nbhd_idx) {

    int64_t batch_size = query.size(0);
    int64_t heads = query.size(1);
    int64_t length = query.size(2);
    int64_t dim = query.size(3);
    int64_t nbhd_size = nbhd_idx.size(2);
    int zsize = batch_size * heads;

    int CHANNELTHREADS = cuda_min(int64_t(CUDA_NUM_THREADS), dim);
    int TOKENTHREADS = cuda_min(int64_t(CUDA_NUM_THREADS / CHANNELTHREADS), length);
    int BATCHTHREADS = max(1, CUDA_NUM_THREADS / (TOKENTHREADS * CHANNELTHREADS));

    auto d_query = torch::zeros_like(query);
    auto d_key = torch::zeros_like(key);

    const auto stream = c10::cuda::getCurrentCUDAStream();

    const dim3 blocks(
            (dim + CHANNELTHREADS - 1) / CHANNELTHREADS,
            (length + TOKENTHREADS - 1) / TOKENTHREADS,
            (zsize + BATCHTHREADS - 1) / BATCHTHREADS);

    const dim3 threads(CHANNELTHREADS, TOKENTHREADS, BATCHTHREADS);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(query.scalar_type(), "clusten_qk_cuda_backward", ([&] {
        const auto d_attn_a = d_attn.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
        const auto query_a = query.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
        const auto key_a = key.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
        const auto nbhd_idx_a = nbhd_idx.packed_accessor32<int64_t,3,torch::DefaultPtrTraits>();
        auto d_query_a = d_query.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
        auto d_key_a = d_key.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();

        const size_t d_key_numel = d_key.numel();
        clusten_qk_cuda_backward_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                d_attn_a, query_a, key_a, nbhd_idx_a, d_query_a, d_key_a,
                length, batch_size, heads, nbhd_size, dim, d_key_numel);
    }));

    return {d_query, d_key.to(key.dtype())};
}