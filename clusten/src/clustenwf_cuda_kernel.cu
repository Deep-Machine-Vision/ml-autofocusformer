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
#include <cuda_fp16.h>
#include <mma.h>

#define CUDA_NUM_THREADS 1024
#define WARP_SIZE 32

template<typename T1, typename T2>
__host__ __device__ constexpr auto cuda_min(T1 a, T2 b) -> typename std::common_type<T1, T2>::type {
    return (a < b) ? static_cast<typename std::common_type<T1, T2>::type>(a) : static_cast<typename std::common_type<T1, T2>::type>(b);
}

template<typename scalar_t>
struct VectorizedLoad {
    using Type = scalar_t;
    static constexpr int VectorSize = 1;
    
    __device__ static Type load(const scalar_t* ptr) {
        return *ptr;
    }
};

template<>
struct VectorizedLoad<float> {
    using Type = float4;
    static constexpr int VectorSize = 4;

    __device__ static Type load(const float* ptr) {
        return *reinterpret_cast<const float4*>(ptr);
    }
};

template<>
struct VectorizedLoad<c10::Half> {
    using Type = float4; // Load 8 halfs as float4
    static constexpr int VectorSize = 8;

    __device__ static Type load(const c10::Half* ptr) {
        return *reinterpret_cast<const float4*>(ptr);
    }
};


template <typename scalar_t, int NBHD_SIZE, int DIM, int DIM_INNER>
__global__ void __launch_bounds__(256, 2) clusten_wf_cuda_forward_kernel_vectorized(
    const torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> weights,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::DefaultPtrTraits> feat,
    const torch::PackedTensorAccessor32<int64_t,3,torch::DefaultPtrTraits> nbhd_idx,
    torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> feat_new,
    const int length, const int length_out, const int batch_size) {

    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int grid_size = gridDim.x * blockDim.x;

    constexpr int VecSize = VectorizedLoad<scalar_t>::VectorSize;
    const int64_t total_work = (int64_t)batch_size * length_out * DIM_INNER;

    for (int64_t work_idx = thread_id; work_idx < total_work; work_idx += grid_size) {
            const int ic = work_idx % DIM_INNER;
            const int64_t bi = work_idx / DIM_INNER;
            const int i = bi % length_out;
            const int b = bi / length_out;

            // Cache neighbor indices
            int64_t neighbors[NBHD_SIZE];
            #pragma unroll
            for (int ni = 0; ni < NBHD_SIZE; ++ni) {
                    neighbors[ni] = nbhd_idx[b][i][ni];
            }

            for (int c_base = 0; c_base < DIM; c_base += VecSize) {
                    scalar_t results[VecSize] = {0};

                    #pragma unroll
                    for (int ni = 0; ni < NBHD_SIZE; ++ni) {
                            const scalar_t weight_val = weights[b][i][ni][ic];

                            if constexpr (VecSize == 1) {
                                    if (c_base < DIM) {
                                            results[0] += weight_val * feat[b][neighbors[ni]][c_base];
                                    }
                            } else if constexpr (VecSize == 4) {
                                    if (c_base + 4 <= DIM) {
                                            auto feat_vec = VectorizedLoad<scalar_t>::load(&feat[b][neighbors[ni]][c_base]);
                                            results[0] += weight_val * feat_vec.x;
                                            results[1] += weight_val * feat_vec.y;
                                            results[2] += weight_val * feat_vec.z;
                                            results[3] += weight_val * feat_vec.w;
                                    } else {
                                            for (int offset = 0; offset < 4 && c_base + offset < DIM; ++offset) {
                                                    results[offset] += weight_val * feat[b][neighbors[ni]][c_base + offset];
                                            }
                                    }
                            } else if constexpr (VecSize == 8) {
                                    if (c_base + 8 <= DIM) {
                                            auto feat_vec = VectorizedLoad<scalar_t>::load(&feat[b][neighbors[ni]][c_base]);
                                            half2* h2_ptr = reinterpret_cast<half2*>(&feat_vec);
                                            #pragma unroll
                                            for (int j = 0; j < 4; ++j) {
                                                    half2 h2_val = h2_ptr[j];
                                                    results[j*2] += weight_val * __half2float(h2_val.x);
                                                    results[j*2+1] += weight_val * __half2float(h2_val.y);
                                            }
                                    } else {
                                        for (int offset = 0; offset < 8 && c_base + offset < DIM; ++offset) {
                                                results[offset] += weight_val * feat[b][neighbors[ni]][c_base + offset];
                                        }
                                    }
                            }
                    }

                    #pragma unroll
                    for (int offset = 0; offset < VecSize && c_base + offset < DIM; ++offset) {
                            feat_new[b][i][ic][c_base + offset] = results[offset];
                    }
            }
    }
}

template <typename scalar_t>
__global__ void __launch_bounds__(256, 2) clusten_wf_cuda_forward_kernel_tiled(
    const torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> weights,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::DefaultPtrTraits> feat,
    const torch::PackedTensorAccessor32<int64_t,3,torch::DefaultPtrTraits> nbhd_idx,
    torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> feat_new,
    const int length, const int length_out, const int batch_size,
    const int nbhd_size, const int dim, const int dim_inner) {

    // Shared memory for tiling
    extern __shared__ char shared_mem[];
    scalar_t* shared_feat = reinterpret_cast<scalar_t*>(shared_mem);

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int block_size = blockDim.x;

    constexpr int TILE_DIM = 64;
    constexpr int TILE_NBHD = 32;

    const int b = bid % batch_size;
    const int i = (bid / batch_size) % length_out;
    const int ic_base = (bid / (batch_size * length_out)) * block_size;

    const int ic = ic_base + tid;

    if (b >= batch_size || i >= length_out || ic >= dim_inner) return;

    int64_t neighbors[96]; // Max neighborhood size
    for (int ni = tid; ni < nbhd_size; ni += block_size) {
        neighbors[ni] = nbhd_idx[b][i][ni];
    }
    __syncthreads();

    for (int c_tile = 0; c_tile < dim; c_tile += TILE_DIM) {
            const int c_end = cuda_min(c_tile + TILE_DIM, dim);

            for (int ni_tile = 0; ni_tile < nbhd_size; ni_tile += TILE_NBHD) {
                    const int ni_end = cuda_min(ni_tile + TILE_NBHD, nbhd_size);

                    for (int idx = tid; idx < (ni_end - ni_tile) * (c_end - c_tile); idx += block_size) {
                            const int local_ni = idx / (c_end - c_tile);
                            const int local_c = idx % (c_end - c_tile);
                            const int ni = ni_tile + local_ni;
                            const int c = c_tile + local_c;

                            if (ni < nbhd_size && c < dim) {
                                    shared_feat[local_ni * TILE_DIM + local_c] = feat[b][neighbors[ni]][c];
                            }
                    }
                    __syncthreads();

                    scalar_t result = 0;
                    for (int c = c_tile; c < c_end; ++c) {
                        for (int ni = ni_tile; ni < ni_end; ++ni) {
                                if (ni < nbhd_size) {
                                        const int local_ni = ni - ni_tile;
                                        const int local_c = c - c_tile;
                                        result += weights[b][i][ni][ic] * shared_feat[local_ni * TILE_DIM + local_c];
                                }
                        }

                        if (ni_tile == 0) {
                                feat_new[b][i][ic][c] = result;
                        } else {
                                feat_new[b][i][ic][c] += result;
                        }
                        result = 0;
                    }
                __syncthreads();
            }
    }
}

template<typename scalar_t>
void launch_specialized_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits>& weights,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::DefaultPtrTraits>& feat,
    const torch::PackedTensorAccessor32<int64_t,3,torch::DefaultPtrTraits>& nbhd_idx,
    torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits>& feat_new,
    int length, int length_out, int batch_size, int nbhd_size, int dim, int dim_inner,
    cudaStream_t stream) {

    const int64_t total_work = batch_size * length_out * dim_inner;
    const int threads = 256;
    const int blocks = cuda_min((int64_t)2048, (total_work + threads - 1) / threads);

    if (nbhd_size == 12 && dim == 32 && dim_inner == 4) {
        clusten_wf_cuda_forward_kernel_vectorized<scalar_t, 12, 32, 4><<<blocks, threads, 0, stream>>>(
            weights, feat, nbhd_idx, feat_new, length, length_out, batch_size);
    } else if (nbhd_size == 24 && dim == 64 && dim_inner == 8) {
        clusten_wf_cuda_forward_kernel_vectorized<scalar_t, 24, 64, 8><<<blocks, threads, 0, stream>>>(
            weights, feat, nbhd_idx, feat_new, length, length_out, batch_size);
    } else if (nbhd_size == 48 && dim == 32 && dim_inner == 4) {
        clusten_wf_cuda_forward_kernel_vectorized<scalar_t, 48, 32, 4><<<blocks, threads, 0, stream>>>(
            weights, feat, nbhd_idx, feat_new, length, length_out, batch_size);
    } else if (nbhd_size == 96 && dim == 32 && dim_inner == 4) {
        clusten_wf_cuda_forward_kernel_vectorized<scalar_t, 96, 32, 4><<<blocks, threads, 0, stream>>>(
            weights, feat, nbhd_idx, feat_new, length, length_out, batch_size);
    } else if (nbhd_size == 12 && dim == 128 && dim_inner == 16) {
        clusten_wf_cuda_forward_kernel_vectorized<scalar_t, 12, 128, 16><<<blocks, threads, 0, stream>>>(
            weights, feat, nbhd_idx, feat_new, length, length_out, batch_size);
    } else {
        const int shared_mem_size = 96 * 64 * sizeof(scalar_t); // Max tile size
        clusten_wf_cuda_forward_kernel_tiled<scalar_t><<<blocks, threads, shared_mem_size, stream>>>(
            weights, feat, nbhd_idx, feat_new, length, length_out, batch_size, nbhd_size, dim, dim_inner);
    }
}

template <typename scalar_t>
__global__ void __launch_bounds__(512, 2) clusten_wf_cuda_backward_kernel_fast(
    const torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> d_feat_new,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> weights,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::DefaultPtrTraits> feat,
    const torch::PackedTensorAccessor32<int64_t,3,torch::DefaultPtrTraits> nbhd_idx,
    torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> d_weights,
    torch::PackedTensorAccessor32<scalar_t,3,torch::DefaultPtrTraits> d_feat,
    const int length, const int length_out, const int batch_size,
    const int nbhd_size, const int dim, const int dim_inner) {

    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int grid_size = gridDim.x * blockDim.x;

    const int64_t total_work_weights = (int64_t)batch_size * length_out * nbhd_size * dim_inner;

    for (int64_t work_idx = thread_id; work_idx < total_work_weights; work_idx += grid_size) {
            const int ic = work_idx % dim_inner;
            const int64_t bini_temp = work_idx / dim_inner;
            const int ni = bini_temp % nbhd_size;
            const int64_t bi_temp = bini_temp / nbhd_size;
            const int i = bi_temp % length_out;
            const int b = bi_temp / length_out;

            const int64_t nbi = nbhd_idx[b][i][ni];

            scalar_t dw_result = 0;

            if (dim == 32) {
                #pragma unroll 32
                for (int c = 0; c < 32; ++c) {
                        dw_result += feat[b][nbi][c] * d_feat_new[b][i][ic][c];
                }
            } else if (dim == 64) {
                #pragma unroll 64
                for (int c = 0; c < 64; ++c) {
                        dw_result += feat[b][nbi][c] * d_feat_new[b][i][ic][c];
                }
            } else if (dim == 128) {
                #pragma unroll 128
                for (int c = 0; c < 128; ++c) {
                        dw_result += feat[b][nbi][c] * d_feat_new[b][i][ic][c];
                }
            } else {
                constexpr int UNROLL = 8;
                int c = 0;
                for (; c + UNROLL - 1 < dim; c += UNROLL) {
                        #pragma unroll
                        for (int offset = 0; offset < UNROLL; ++offset) {
                                dw_result += feat[b][nbi][c + offset] * d_feat_new[b][i][ic][c + offset];
                        }
                }
                for (; c < dim; ++c) {
                        dw_result += feat[b][nbi][c] * d_feat_new[b][i][ic][c];
                }
            }

            d_weights[b][i][ni][ic] = dw_result;
    }

    __syncthreads();

    const int64_t total_work_feat = (int64_t)batch_size * length * dim;

    for (int64_t work_idx = thread_id; work_idx < total_work_feat; work_idx += grid_size) {
            const int c = work_idx % dim;
            const int64_t b_nbi_temp = work_idx / dim;
            const int nbi = b_nbi_temp % length;
            const int b = b_nbi_temp / length;

            scalar_t df_accumulator = 0;

            for (int i = 0; i < length_out; ++i) {
                    for (int ni = 0; ni < nbhd_size; ++ni) {
                            if (nbhd_idx[b][i][ni] == nbi) {
                                    if (dim_inner <= 16) {
                                            #pragma unroll 16
                                            for (int ic = 0; ic < dim_inner; ++ic) {
                                                    df_accumulator += weights[b][i][ni][ic] * d_feat_new[b][i][ic][c];
                                            }
                                    } else {
                                            for (int ic = 0; ic < dim_inner; ++ic) {
                                                    df_accumulator += weights[b][i][ni][ic] * d_feat_new[b][i][ic][c];
                                            }
                                    }
                            }
                    }
            }

            if (df_accumulator != 0) {
                    atomicAdd(&d_feat[b][nbi][c], df_accumulator);
            }
    }
}

torch::Tensor clusten_wf_cuda_forward_opt(
    const torch::Tensor &weights,
    const torch::Tensor &feat,
    const torch::Tensor &nbhd_idx) {

    int64_t batch_size = weights.size(0);
    int64_t length_out = weights.size(1);
    int64_t nbhd_size = weights.size(2);
    int64_t dim_inner = weights.size(3);
    int64_t length = feat.size(1);
    int64_t dim = feat.size(2);

    auto feat_new = torch::zeros({batch_size, length_out, dim_inner, dim}, weights.options());

    const auto stream = c10::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(weights.scalar_type(), "clusten_wf_cuda_forward_opt", ([&] {
            const auto weights_a = weights.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
            const auto feat_a = feat.packed_accessor32<scalar_t,3,torch::DefaultPtrTraits>();
            const auto nbhd_idx_a = nbhd_idx.packed_accessor32<int64_t,3,torch::DefaultPtrTraits>();
            auto feat_new_a = feat_new.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();

            launch_specialized_forward_kernel<scalar_t>(
                weights_a, feat_a, nbhd_idx_a, feat_new_a,
                length, length_out, batch_size, nbhd_size, dim, dim_inner, stream);
    }));

    return feat_new;
}

std::vector<torch::Tensor> clusten_wf_cuda_backward_opt(
    const torch::Tensor &d_feat_new,
    const torch::Tensor &weights,
    const torch::Tensor &feat,
    const torch::Tensor &nbhd_idx) {

    int64_t batch_size = weights.size(0);
    int64_t length_out = weights.size(1);
    int64_t nbhd_size = weights.size(2);
    int64_t dim_inner = weights.size(3);
    int64_t length = feat.size(1);
    int64_t dim = feat.size(2);

    auto d_weights = torch::zeros_like(weights);
    auto d_feat = torch::zeros_like(feat);

    const auto stream = c10::cuda::getCurrentCUDAStream();

    const int threads_per_block = 512;
    const int64_t max_work = cuda_min(
                                        (int64_t)batch_size * length_out * nbhd_size * dim_inner,
                                        (int64_t)batch_size * length * dim
                                    );
    const int blocks = cuda_min((int64_t)4096, (max_work + threads_per_block - 1) / threads_per_block);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(weights.scalar_type(), "clusten_wf_cuda_backward_opt", ([&] {
            const auto d_feat_new_a = d_feat_new.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
            const auto weights_a = weights.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
            const auto feat_a = feat.packed_accessor32<scalar_t,3,torch::DefaultPtrTraits>();
            const auto nbhd_idx_a = nbhd_idx.packed_accessor32<int64_t,3,torch::DefaultPtrTraits>();
            auto d_weights_a = d_weights.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
            auto d_feat_a = d_feat.packed_accessor32<scalar_t,3,torch::DefaultPtrTraits>();

            clusten_wf_cuda_backward_kernel_fast<scalar_t><<<blocks, threads_per_block, 0, stream>>>(
                d_feat_new_a, weights_a, feat_a, nbhd_idx_a, d_weights_a, d_feat_a,
                length, length_out, batch_size, nbhd_size, dim, dim_inner);
    }));

    return {d_weights, d_feat.to(feat.dtype())};
}

template <typename scalar_t>
__global__ void clusten_wf_cuda_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> weights,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::DefaultPtrTraits> feat,
    const torch::PackedTensorAccessor32<int64_t,3,torch::DefaultPtrTraits> nbhd_idx,
    torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> feat_new,
    const int length, const int length_out, const int batch_size,
    const int nbhd_size, const int dim, const int dim_inner) {

    const int b = blockIdx.z * blockDim.z + threadIdx.z;
    if (b < batch_size){
        const int i = blockIdx.y * blockDim.y + threadIdx.y;
        if (i < length_out){
            const int c = blockIdx.x * blockDim.x + threadIdx.x;
            if (c < dim){
                int64_t nbi;
                scalar_t updt;
                #pragma unroll
                for (unsigned int ic=0; ic < dim_inner; ++ic) {
                    updt = 0;
                    #pragma unroll
                    for (unsigned int ni=0; ni < nbhd_size; ++ni) {
                        nbi = nbhd_idx[b][i][ni];
                        updt += weights[b][i][ni][ic] * feat[b][nbi][c];
                    }
                    feat_new[b][i][ic][c] = updt;
                }
            }
        }
    }
}

torch::Tensor clusten_wf_cuda_forward(
    const torch::Tensor &weights,
    const torch::Tensor &feat,
    const torch::Tensor &nbhd_idx) {

    int64_t batch_size = weights.size(0);
    int64_t length_out = weights.size(1);
    int64_t nbhd_size = weights.size(2);
    int64_t dim_inner = weights.size(3);
    int64_t length = feat.size(1);
    int64_t dim = feat.size(2);

    int CHANNELTHREADS = cuda_min(int64_t(CUDA_NUM_THREADS), dim);
    int TOKENTHREADS = cuda_min(int64_t(CUDA_NUM_THREADS / CHANNELTHREADS), length_out);
    int BATCHTHREADS = max(1, CUDA_NUM_THREADS / (TOKENTHREADS * CHANNELTHREADS));

    auto feat_new = torch::zeros(
        {batch_size, length_out, dim_inner, dim}, weights.options());

    const auto stream = c10::cuda::getCurrentCUDAStream();
    const dim3 blocks(
        (dim + CHANNELTHREADS - 1) / CHANNELTHREADS,
        (length_out + TOKENTHREADS - 1) / TOKENTHREADS,
        (batch_size + BATCHTHREADS - 1) / BATCHTHREADS);
    const dim3 threads(CHANNELTHREADS, TOKENTHREADS, BATCHTHREADS);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(weights.scalar_type(), "clusten_wf_cuda_forward", ([&] {
        const auto weights_a = weights.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
        const auto feat_a = feat.packed_accessor32<scalar_t,3,torch::DefaultPtrTraits>();
        const auto nbhd_idx_a = nbhd_idx.packed_accessor32<int64_t,3,torch::DefaultPtrTraits>();
        auto feat_new_a = feat_new.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();

        clusten_wf_cuda_forward_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            weights_a, feat_a, nbhd_idx_a, feat_new_a,
            length, length_out, batch_size, nbhd_size, dim, dim_inner);
    }));
    return feat_new;
}

template <typename scalar_t>
__global__ void clusten_wf_cuda_backward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> d_feat_new,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> weights,
    const torch::PackedTensorAccessor32<int64_t,3,torch::DefaultPtrTraits> nbhd_idx,
    torch::PackedTensorAccessor32<scalar_t,3,torch::DefaultPtrTraits> d_feat,
    const int length, const int length_out, const int batch_size,
    const int nbhd_size, const int dim, const int dim_inner,
    const size_t d_feat_numel) {

    const int b = blockIdx.z * blockDim.z + threadIdx.z;
    if (b < batch_size){
        const int i = blockIdx.y * blockDim.y + threadIdx.y;
        if (i < length_out){
            const int c = blockIdx.x * blockDim.x + threadIdx.x;
            if (c < dim){
                int64_t nbi;
                size_t index;
                scalar_t updt;
                #pragma unroll
                for (unsigned int ni=0; ni < nbhd_size; ++ni) {
                    nbi = nbhd_idx[b][i][ni];
                    updt = 0;
                    #pragma unroll
                    for (unsigned int ic=0; ic < dim_inner; ++ic) {
                        updt += d_feat_new[b][i][ic][c] * weights[b][i][ni][ic];
                    }
                    index = b*d_feat.stride(0) + nbi*d_feat.stride(1) + c;
                    at::native::fastAtomicAdd(d_feat.data(), index, d_feat_numel, updt, true);
                }
            }
        }
    }
}

template <typename scalar_t>
__global__ void clusten_wf_weights_cuda_backward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> d_feat_new,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::DefaultPtrTraits> feat,
    const torch::PackedTensorAccessor32<int64_t,3,torch::DefaultPtrTraits> nbhd_idx,
    torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> d_weights,
    const int length, const int length_out, const int batch_size,
    const int nbhd_size, const int dim, const int dim_inner){

    const int b = blockIdx.z * blockDim.z + threadIdx.z;
    if (b < batch_size){
        const int i = blockIdx.y * blockDim.y + threadIdx.y;
        if (i < length_out){
            const int z = blockIdx.x * blockDim.x + threadIdx.x;
            if (z < nbhd_size * dim_inner){
                const int ni = z / dim_inner;
                const int ic = z - ni * dim_inner;
                int64_t nbi = nbhd_idx[b][i][ni];
                scalar_t updt = 0;
                #pragma unroll
                for (unsigned int c=0; c < dim; ++c) {
                    updt += feat[b][nbi][c] * d_feat_new[b][i][ic][c];
                }
                d_weights[b][i][ni][ic] = updt;
            }
        }
    }
}

std::vector<torch::Tensor> clusten_wf_cuda_backward(
    const torch::Tensor &d_feat_new,
    const torch::Tensor &weights,
    const torch::Tensor &feat,
    const torch::Tensor &nbhd_idx) {

    int64_t batch_size = weights.size(0);
    int64_t length_out = weights.size(1);
    int64_t nbhd_size = weights.size(2);
    int64_t dim_inner = weights.size(3);
    int64_t length = feat.size(1);
    int64_t dim = feat.size(2);

    int64_t zsize = nbhd_size * dim_inner;

    int CHANNELTHREADS = cuda_min(int64_t(CUDA_NUM_THREADS), dim);
    int TOKENTHREADS = cuda_min(int64_t(CUDA_NUM_THREADS / CHANNELTHREADS), length_out);
    int BATCHTHREADS = max(1, CUDA_NUM_THREADS / (TOKENTHREADS * CHANNELTHREADS));

    int NBHDTHREADS = cuda_min(int64_t(CUDA_NUM_THREADS), zsize);
    int TOKENTHREADS_NB = cuda_min(int64_t(CUDA_NUM_THREADS / NBHDTHREADS), length_out);
    int BATCHTHREADS_NB = max(1, CUDA_NUM_THREADS / (TOKENTHREADS_NB * NBHDTHREADS));

    auto d_weights = torch::zeros_like(weights);
    auto d_feat = torch::zeros_like(feat);

    const auto stream = c10::cuda::getCurrentCUDAStream();

    const dim3 blocks(
        (dim + CHANNELTHREADS - 1) / CHANNELTHREADS,
        (length_out + TOKENTHREADS - 1) / TOKENTHREADS,
        (batch_size + BATCHTHREADS - 1) / BATCHTHREADS);
    const dim3 threads(CHANNELTHREADS, TOKENTHREADS, BATCHTHREADS);

    const dim3 blocks_nb(
        (zsize + NBHDTHREADS - 1) / NBHDTHREADS,
        (length_out + TOKENTHREADS_NB - 1) / TOKENTHREADS_NB,
        (batch_size + BATCHTHREADS_NB - 1) / BATCHTHREADS_NB);
    const dim3 threads_nb(NBHDTHREADS, TOKENTHREADS_NB, BATCHTHREADS_NB);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(weights.scalar_type(), "clusten_wf_cuda_backward", ([&] {
        const auto d_feat_new_a = d_feat_new.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
        const auto weights_a = weights.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
        const auto feat_a = feat.packed_accessor32<scalar_t,3,torch::DefaultPtrTraits>();
        const auto nbhd_idx_a = nbhd_idx.packed_accessor32<int64_t,3,torch::DefaultPtrTraits>();
        auto d_weights_a = d_weights.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
        auto d_feat_a = d_feat.packed_accessor32<scalar_t,3,torch::DefaultPtrTraits>();

        const size_t d_feat_numel = d_feat.numel();
        clusten_wf_cuda_backward_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            d_feat_new_a, weights_a, nbhd_idx_a, d_feat_a,
            length, length_out, batch_size, nbhd_size, dim, dim_inner, d_feat_numel);
        clusten_wf_weights_cuda_backward_kernel<scalar_t><<<blocks_nb, threads_nb, 0, stream>>>(
            d_feat_new_a, feat_a, nbhd_idx_a, d_weights_a,
            length, length_out, batch_size, nbhd_size, dim, dim_inner);
    }));

    return {d_weights, d_feat.to(feat.dtype())};
}