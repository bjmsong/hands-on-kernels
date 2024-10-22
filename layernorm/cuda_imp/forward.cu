/*
Compile example:
nvcc -O3 --use_fast_math -lcublas -lcublasLt forward.cu -o layernorm_forward

For Compute Sanitizer
nvcc -lineinfo --use_fast_math -lcublas -lcublasLt forward.cu -o layernorm_forward

For Debug
nvcc -g -G --use_fast_math -lcublas -lcublasLt forward.cu -o layernorm_forward
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <assert.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include "../gpu/helper.h"

// ----------------------------------------------------------------------------
// CPU code reference

// GPT-2 layernorm forward pass
void layernorm_forward_cpu(float* out, float* mean, float* rstd,
                       const float* inp, const float* weight, const float* bias,
                       int B, int T, int C) {
    float eps = 1e-5f;
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            // seek to the input position inp[b,t,:]
            const float* x = inp + b * T * C + t * C;

            // calculate the mean
            float m = 0.0f;
            for (int i = 0; i < C; i++) {
                m += x[i];
            }
            m = m/C;

            // calculate the variance (without any bias correction)
            float v = 0.0f;
            for (int i = 0; i < C; i++) {
                float xshift = x[i] - m;
                v += xshift * xshift;
            }
            v = v/C;

            // calculate the rstd
            float s = 1.0f / sqrtf(v + eps);

            // seek to the output position in out[b,t,:]
            float* out_bt = out + b * T * C + t * C;

            // calculate the output
            for (int i = 0; i < C; i++) {
                float n = (s * (x[i] - m)); // normalized output
                float o = n * weight[i] + bias[i]; // scale and shift it
                out_bt[i] = o; // write
            }
            // cache the mean and rstd for the backward pass later
            mean[b * T + t] = m;
            rstd[b * T + t] = s;
        }
    }
}

// ----------------------------------------------------------------------------
// GPU kernels
__global__ void layernorm_forward_kernel(float* out, const float* inp, float* mean, float* rstd, const float* weight, 
                                 const float* bias, int C) {
    int idx = blockIdx.x;   // [0, B*T)
    int tid = threadIdx.x;  // [0, block_size)
    const float* x = inp + idx * C;
    out = out + idx * C;

    // 1. calculate mean
    extern __shared__ float shared[];  // size is decided during kernel launch
    // thread coarsening
    float sum = 0.0f;
    for (int i = tid; i < C; i += blockDim.x) {
        sum += x[i];
    }
    shared[tid] = sum;
    __syncthreads();
    // reduction
    for (int stride = blockDim.x / 2; stride >= 1; stride /= 2) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }
    __shared__ float avg;
    if (tid == 0) {
        avg = shared[tid]/C;        
    }
    __syncthreads();
    
    // 2. calculate rstd
   // thread coarsening
    sum = 0.0f;
    for (int i = tid; i < C; i += blockDim.x) {
        float diff = x[i] - avg;
        sum += diff * diff;
    }
    shared[tid] = sum;
    __syncthreads();
    // reductions
    for (int stride = blockDim.x / 2; stride >= 1; stride /= 2) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }
    __shared__ float Rstd;
    if (tid == 0) {
        Rstd  = 1.0f / sqrtf(shared[0] / C + 1e-5f);
    }
    __syncthreads();
    
    // 3. norm input, write to output
    for (int i = tid; i < C; i += blockDim.x) {
        out[i] = (x[i] - avg) * Rstd * weight[i] + bias[i];
    }
}

void layernorm_forward(float* out, float* mean, float* rstd,
                           const float* inp, const float* weight, const float* bias,
                           int B, int T, int C,
                           const int block_size) {
    const int N = B * T;
    layernorm_forward_kernel<<<N, block_size, block_size * sizeof(float)>>>(out, inp, mean, rstd, weight, bias, C);
    checkCuda(cudaGetLastError());
}


// ----------------------------------------------------------------------------

int main(int argc, char **argv) {
    srand(0);

    int B = 8;
    int T = 1024;  // sequence length
    int C = 768;   // channels, feature size

    int deviceIdx = 0;
    checkCuda(cudaSetDevice(deviceIdx));

    // create host memory of random numbers
    float* out = (float*)malloc(B * T * C * sizeof(float));
    float* mean = (float*)malloc(B * T * sizeof(float));
    float* rstd = (float*)malloc(B * T * sizeof(float));
    float* inp = make_random_float(B * T * C);
    float* weight = make_random_float(C);
    float* bias = make_random_float(C);

    // move to GPU
    float* d_out;
    float* d_mean;
    float* d_rstd;
    float* d_inp;
    float* d_weight;
    float* d_bias;
    checkCuda(cudaMalloc(&d_out, B * T * C * sizeof(float)));
    checkCuda(cudaMalloc(&d_mean, B * T * sizeof(float)));
    checkCuda(cudaMalloc(&d_rstd, B * T * sizeof(float)));
    checkCuda(cudaMalloc(&d_inp, B * T * C * sizeof(float)));
    checkCuda(cudaMalloc(&d_weight, C * sizeof(float)));
    checkCuda(cudaMalloc(&d_bias, C * sizeof(float)));
    checkCuda(cudaMemcpy(d_inp, inp, B * T * C * sizeof(float), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_weight, weight, C * sizeof(float), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_bias, bias, C * sizeof(float), cudaMemcpyHostToDevice));

    int block_sizes[] = {64, 128, 256, 512, 1024};

    layernorm_forward_cpu(out, mean, rstd, inp, weight, bias, B, T, C);

    // check the correctness of the kernel at all block sizes
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];
        printf("Checking block size %d.\n", block_size);

        layernorm_forward(d_out, d_mean, d_rstd, d_inp, d_weight, d_bias, B, T, C, block_size);

        validate_result(d_out, out, "out", B * T * C, 1e-5f);
        // validate_result(d_mean, mean, "mean", B * T, 1e-5f);
        // validate_result(d_rstd, rstd, "rstd", B * T, 1e-5f);
    }

    printf("All results match. Starting benchmarks.\n\n");

    // time the kernel at different block sizes
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];

        int repeat_times = 2000;
        float elapsed_time = benchmark_kernel(repeat_times, layernorm_forward,
                                              d_out, d_mean, d_rstd, d_inp, d_weight, d_bias,
                                              B, T, C, block_size);

        // napkin math: estimate the memory bandwidth achieved
        // e.g. A100 40GB PCIe is advertised at 1,555GB/s
        long memory_ops = (2 * B * T * C) * 4; // *4 for float
        float memory_bandwidth = memory_ops / elapsed_time / 1e6;

        /*
        calculate mean：（求和）BTC + （平均）BT 
        calculate variance: （减去均值+求平方+求和）3BTC + （求平均+开根号）2BT  
        normalizaiton：(减去均值+除以标准差)2BTC
        缩放和偏移：（乘以权重+加上偏置）2BTC
        */
        long flop = (B * T * C) * 8 + (B * T) * 3;
        float flops = flop / elapsed_time / 1e9;

        printf("block_size %4d | time %.4f ms | bandwidth %.2f GB/s | FLOPS %.2f TFLOPS \n", block_size, elapsed_time, memory_bandwidth, flops);
    }

    // free memory
    free(out);
    free(mean);
    free(rstd);
    free(inp);
    free(weight);
    free(bias);
    checkCuda(cudaFree(d_out));
    checkCuda(cudaFree(d_mean));
    checkCuda(cudaFree(d_rstd));
    checkCuda(cudaFree(d_inp));
    checkCuda(cudaFree(d_weight));
    checkCuda(cudaFree(d_bias));

    return 0;
}