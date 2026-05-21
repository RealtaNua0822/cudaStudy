#include <stdio.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 32

// Shared Memory 矩阵乘法核函数
__global__ void matrixMulShared(float *A, float *B, float *C, int N) {
    // 1. 申请线程块内部共享的二级缓存瓷砖
    // 这里的空间是驻留在 SM 内部的，每个 Block 独享一份
    __shared__ float s_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float s_B[BLOCK_SIZE][BLOCK_SIZE];

    int bx = blockIdx.x;  int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    // 当前线程负责计算的 C 矩阵绝对坐标
    int row = by * BLOCK_SIZE + ty;
    int col = bx * BLOCK_SIZE + tx;

    // 累加寄存器，暂存当前线程的计算结果
    float sum = 0.0f;

    // 2. 开始沿着 K 轴以 BLOCK_SIZE（32）为步长分批迭代
    for (int ph = 0; ph < N / BLOCK_SIZE; ++ph) {
        
        // 3. 集体搬运：协同将 Global Memory 数据加载到 Shared Memory 瓷砖中
        // 这里的物理访存会被合并（Coalesced），效率极高
        s_A[ty][tx] = A[row * N + (ph * BLOCK_SIZE + tx)];
        s_B[ty][tx] = B[(ph * BLOCK_SIZE + ty) * N + col];

        // 4. 第一次同步：必须等待 Block 内所有线程都把格子填满
        __syncthreads();

        // 5. 瓷砖内部计算：线程在极快的 Shared Memory 里进行 32 次点积累加
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += s_A[ty][k] * s_B[k][tx];
        }

        // 6. 第二次同步：确保所有线程都算完了，再进入下一轮循环去覆盖 s_A 和 s_B
        __syncthreads();
    }

    // 7. 把最终结果写回全局显存 C
    C[row * N + col] = sum;
}

// 主函数：包含数据初始化、硬件耗时统计与验证
int main() {
    int N = 4096;
    size_t size = N * N * sizeof(float);

    // 分配主机内存
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    // 初始化数据（为了验证方便，A全设为1.0f，B全设为2.0f）
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // 分配设备显存
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // 拷贝到显存
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // 定义网格与线程块
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks(N / BLOCK_SIZE, N / BLOCK_SIZE);

    printf("🚀 RTX 4060 开始冲击 Shared Memory 4096 算子...\n");

    // 创建 CUDA Event 统计纯粹的硬件执行时间
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 预热核函数（排除懒加载干扰）
    matrixMulShared<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    // 正式计时开始
    cudaEventRecord(start);
    matrixMulShared<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    float msecTotal = 0;
    cudaEventElapsedTime(&msecTotal, start, stop);

    // 计算吞吐量 (GFLOPS)
    // 4096^3 * 2 代表：4096*4096*4096 次乘法 + 4096*4096*4096 次加法
    double flopsPerMatrixMul = 2.0 * N * N * N;
    double gflops = (flopsPerMatrixMul * 1.0e-9) / (msecTotal / 1000.0);

    // 拷贝回主机验证
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    printf("⏱️ 真实耗时: %.2f 毫秒\n", msecTotal);
    printf("📊 实际性能: %.3f GFLOPS\n", gflops);
    printf("🔍 验证坐标[0][0]结果: %.0f (预期值: %d)\n", h_C[0], N * 2);

    // 释放资源
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}