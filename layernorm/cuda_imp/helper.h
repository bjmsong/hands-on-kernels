#include <iostream>
#include <random>
#include <cstdio>
#include <float.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

cublasHandle_t cublas_handle;
std::default_random_engine random_engine(0);
std::uniform_real_distribution<float> uniform_dist(-256, 256);

#define checkCuda(val) check((val), #val, __FILE__, __LINE__)
void check(cudaError_t err, const char* const func, const char* const file,
           const int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define checkcuBLAS(val) checkcublas((val), #val, __FILE__, __LINE__)
void checkcublas(cublasStatus_t status, const char* const func, const char* const file,
           const int line)
{
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << "[cuBLAS ERROR]: " << status << " " << file << " " << line << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

void matrix_init(float* a, int M, int N){
    // A(M,N)
    for(int i=0; i<M; i++){
        for(int j=0; j<N; j++){
            a[i*N + j] = uniform_dist(random_engine);
        }
    }
}

void matrix_init(half* a, int M, int N){
    // A(M,N)
    for(int i=0; i<M; i++){
        for(int j=0; j<N; j++){
            a[i*N + j] = __float2half(uniform_dist(random_engine));
        }
    }
}

bool FloatEqual(float a, float b, float epsilon = 1e-4) {
    return std::abs(a - b) < epsilon;
}

static void PrintAssert(bool condition, float a, float b) {
    if (!condition) {
        printf("%f is not equal to %f\n", a, b);
    }
}

void checkResult(float* a, float* b, float* c, size_t bytes, int M, int N, int K){

    float* c_check;
    checkCuda(cudaMalloc(&c_check, bytes));

    cublasHandle_t handle;
	cublasCreate(&handle);
	float alpha = 1.0f;
	float beta = 0.0f;
    // Calculate: c = (alpha*a) * b + (beta*c)
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, K, M, N,  
    &alpha, b, K, a, N, &beta, c_check, K);

    float* h_c_check = (float*)malloc(bytes);
    checkCuda(cudaMemcpy(h_c_check, c_check, bytes, cudaMemcpyDeviceToHost));

    for (int i=0; i < M; i++){
        for (int j=0; j < K; j++)
            PrintAssert(FloatEqual(c[i*K + j],h_c_check[i*K + j]), c[i*K + j], h_c_check[i*K + j]);
    }

    free(h_c_check);
    checkCuda(cudaFree(c_check));
}

void checkResult(half* a, half* b, float* c, size_t bytes, int M, int N, int K){

    float* c_check;
    checkCuda(cudaMalloc(&c_check, bytes));

    cublasHandle_t handle;
	cublasCreate(&handle);
	float alpha = 1.0f;
	float beta = 0.0f;
	cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, K, M, N,
    &alpha, b, CUDA_R_16F, K, a, CUDA_R_16F, N, &beta, c_check, CUDA_R_32F, K, CUDA_R_32F,
    static_cast<cublasGemmAlgo_t>(CUBLAS_GEMM_DEFAULT));

    float* h_c_check = (float*)malloc(bytes);
    checkCuda(cudaMemcpy(h_c_check, c_check, bytes, cudaMemcpyDeviceToHost));

    for (int i=0; i < M; i++){
        for (int j=0; j < K; j++)
            PrintAssert(FloatEqual(c[i*K + j],h_c_check[i*K + j]),
          c[i*K + j], h_c_check[i*K + j]);
    }

    free(h_c_check);
    checkCuda(cudaFree(c_check));
}

template<class Kernel, class... KernelArgs>
float benchmark_kernel(int repeats, Kernel kernel, KernelArgs&&... kernel_args) {
    cudaEvent_t start, stop;
    // prepare buffer to scrub L2 cache between benchmarks
    // just memset a large dummy array, recommended by
    // https://stackoverflow.com/questions/31429377/how-can-i-clear-flush-the-l2-cache-and-the-tlb-of-a-gpu
    // and apparently used in nvbench.
    int deviceIdx = 0;
    checkCuda(cudaSetDevice(deviceIdx));
    cudaDeviceProp deviceProp;
    checkCuda(cudaGetDeviceProperties(&deviceProp, deviceIdx));
    void* flush_buffer;
    checkCuda(cudaMalloc(&flush_buffer, deviceProp.l2CacheSize));

    checkCuda(cudaEventCreate(&start));
    checkCuda(cudaEventCreate(&stop));
    float elapsed_time = 0.f;
    for (int i = 0; i < repeats; i++) {
        // clear L2
        checkCuda(cudaMemset(flush_buffer, 0, deviceProp.l2CacheSize));
        // now we can start recording the timing of the kernel
        checkCuda(cudaEventRecord(start, nullptr));
        kernel(std::forward<KernelArgs>(kernel_args)...);
        checkCuda(cudaEventRecord(stop, nullptr));
        checkCuda(cudaEventSynchronize(start));
        checkCuda(cudaEventSynchronize(stop));
        float single_call;
        checkCuda(cudaEventElapsedTime(&single_call, start, stop));
        elapsed_time += single_call;
    }

    checkCuda(cudaFree(flush_buffer));

    return elapsed_time / repeats;
}

template<class D, class T>
void validate_result(D* device_result, const T* cpu_reference, const char* name, std::size_t num_elements, T tolerance=1e-4) {
    D* out_gpu = (D*)malloc(num_elements * sizeof(D));
    checkCuda(cudaMemcpy(out_gpu, device_result, num_elements * sizeof(D), cudaMemcpyDeviceToHost));
    int nfaults = 0;
#ifndef ENABLE_BF16
    float epsilon = FLT_EPSILON;
#else
    float epsilon = 0.079;
#endif
    for (int i = 0; i < num_elements; i++) {
        // Skip masked elements
        if(!isfinite(cpu_reference[i]))
            continue;

        // print the first few comparisons
        if (i < 5) {
            printf("%f %f\n", cpu_reference[i], (T)out_gpu[i]);
        }
        // effective tolerance is based on expected rounding error (epsilon),
        // plus any specified additional tolerance
        float t_eff = tolerance + fabs(cpu_reference[i]) * epsilon;
        // ensure correctness for all elements.
        if (fabs(cpu_reference[i] - (T)out_gpu[i]) > t_eff) {
            printf("Mismatch of %s at %d: CPU_ref: %f vs GPU: %f\n", name, i, cpu_reference[i], (T)out_gpu[i]);
            nfaults ++;
            if (nfaults >= 10) {
                free(out_gpu);
                exit(EXIT_FAILURE);
            }
        }
    }

    if (nfaults > 0) {
        free(out_gpu);
        exit(EXIT_FAILURE);
    }

    free(out_gpu);
}

float* make_random_float(size_t N) {
    float* arr = (float*)malloc(N * sizeof(float));
    for (size_t i = 0; i < N; i++) {
        arr[i] = ((float)rand() / RAND_MAX) * 2.0 - 1.0; // range -1..1
        // arr[i] = 1;  // for debug
    }
    return arr;
}

float* make_zeros_float(size_t N) {
    float* arr = (float*)malloc(N * sizeof(float));
    memset(arr, 0, N * sizeof(float)); // all zero
    return arr;
}

float* make_ones_float(size_t N) {
    float* arr = (float*)malloc(N * sizeof(float));
    for (size_t i = 0; i < N; i++) {
        arr[i] = 1.0f;
    }
    return arr;
}
