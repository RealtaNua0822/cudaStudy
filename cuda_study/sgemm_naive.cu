#include <iostream>
#include <cuda_runtime.h>

// 朴素版（Naive）矩阵乘法核函数
__global__ void matrixMulNaive(float *A, float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f; // 驻留在寄存器
        
        // K轴大循环：4096次！每次迭代都去慢如蜗牛的 Global Memory 捞两个数
        for (int i = 0; i < N; ++i) {
            sum += A[row * N + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

int main() {
    int N = 4096; // 💥 轰到 4096 x 4096 规模，单张矩阵占 64MB，总共 192MB
    size_t size = N * N * sizeof(float);

    // 分配主机内存 (CPU)
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    // 初始化矩阵
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // 分配设备显存 (GPU 4060)
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // 拷贝数据到显存
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // 依然是你的黄金编制：每个 Block 1024 个线程 (32x32)
    dim3 blockDim(32, 32); 
    // 4096 / 32 = 128。所以网格是 128x128 = 16384 个 Block
    dim3 gridDim(N / 32, N / 32); 

    // 使用 CUDA Event 硬件级精准计时
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    std::cout << "🚀 RTX 4060 开始疯狂压榨 4096 Naive 算子..." << std::endl;
    
    cudaEventRecord(start);
    matrixMulNaive<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // 捞回计算结果
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // 算一下实际跑出来的 GFLOPS（每秒十亿次浮点运算）
    // 4096^3 * 2 (一次乘，一次加)
    double flop = 2.0 * N * N * N;
    double gflops = (flop / (milliseconds / 1000.0)) / 1e9;

    std::cout << "🎉 计算完成！" << std::endl;
    std::cout << "⏱️ 真实耗时: " << milliseconds << " 毫秒" << std::endl;
    std::cout << "📊 实际性能: " << gflops << " GFLOPS" << std::endl;
    std::cout << "🔍 验证坐标[0][0]结果: " << h_C[0] << " (预期值: 8192)" << std::endl;

    // 释放
    free(h_A); free(h_B); free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}