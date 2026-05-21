#include <stdio.h>
#include <cuda_runtime.h>

// 宏定义：每个线程计算 8x8 的子块
#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8

__global__ void matrixMulTiling(float *A, float *B, float *C, int N) {
    // 1. 申请大号的 Shared Memory 瓷砖
    __shared__ float s_A[BK][BM]; // 8 x 128
    __shared__ float s_B[BK][BN]; // 8 x 128

    // 2. 线程在当前 Block 内部的几何坐标 (0 到 15)
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * 16 + tx; // 当前线程的绝对一维 ID (0 到 255)

    // 3. 线程私有的寄存器堆，存结果和临时数
    float accum[TM][TN] = {0.0f};
    float reg_A[TM];
    float reg_B[TN];

    // 4. 计算当前线程负责的 C 矩阵左上角基准物理坐标
    int c_row_base = blockIdx.y * BM + ty * TM;
    int c_col_base = blockIdx.x * BN + tx * TN;

    // 5. 计算当前线程在搬运 Global->Shared 时的分工坐标
    // 一个 Block 有 256 个线程，要搬运 s_A (8*128=1024个元素)
    // 也就是说每个线程在一轮中需要搬运 4 个 A 的元素和 4 个 B 的元素
    int a_load_row = tid / (BM / 4); // 1024个点横向展开
    int a_load_col = (tid % (BM / 4)) * 4;
    int b_load_row = tid / (BN / 4);
    int b_load_col = (tid % (BN / 4)) * 4;

    // 开始大循环
    for (int ph = 0; ph < N / BK; ++ph) {
        
        // 6. 向量化并行搬运：利用 float4 一次性搬运 4 个 float（128位宽度），榨干显存带宽
        // 搬运 A 矩阵
        int g_a_row = blockIdx.y * BM + a_load_col;
        int g_a_col = ph * BK + a_load_row;
        // 伪装成 float4 写入，避开单字节写入瓶颈
        #pragma unroll
        for(int i=0; i<4; ++i) {
            s_A[a_load_row][a_load_col + i] = A[(g_a_row + i) * N + g_a_col];
        }

        // 搬运 B 矩阵
        int g_b_row = ph * BK + b_load_row;
        int g_b_col = blockIdx.x * BN + b_load_col;
        *(float4*)&s_B[b_load_row][b_load_col] = *(float4*)&B[g_b_row * N + g_b_col];

        // 严格同步：等待全员搬运完毕
        __syncthreads();

        // 7. 核心寄存器计算：在 BK(8) 的长度内，分步把数拉进寄存器复用
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            // 将 A 和 B 的片段拉进寄存器
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                reg_A[i] = s_A[k][ty * TM + i];
            }
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                reg_B[j] = s_B[k][tx * TN + j];
            }

            // 在寄存器内部进行 8x8 的全外积乘加（完全没有缓存延迟！）
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    accum[i][j] += reg_A[i] * reg_B[j];
                }
            }
        }

        // 再次同步：防止走得快的线程提前改写了 Shared Memory
        __syncthreads();
    }

    // 8. 最终计算完毕，将全套寄存器里憋出来的 64 个结果倾泻回全局显存 C
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            C[(c_row_base + i) * N + (c_col_base + j)] = accum[i][j];
        }
    }
}

int main() {
    int N = 4096;
    size_t size = N * N * sizeof(float);

    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    for (int i = 0; i < N * N; ++i) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // 重新划定网格：Block 内 16x16 线程，Grid 变为原来的 1/4
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(N / BN, N / BM);

    printf("🚀 RTX 4060 正在祭出 Thread Tiling 终极绝活...\n");

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 预热
    matrixMulTiling<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    matrixMulTiling<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    float msecTotal = 0;
    cudaEventElapsedTime(&msecTotal, start, stop);

    double flopsPerMatrixMul = 2.0 * N * N * N;
    double gflops = (flopsPerMatrixMul * 1.0e-9) / (msecTotal / 1000.0);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    printf("⏱️ Tiling 耗时: %.2f 毫秒\n", msecTotal);
    printf("🔥 Tiling 性能: %.3f GFLOPS\n", gflops);
    printf("🔍 验证坐标[0][0]结果: %.0f (预期值: %d)\n", h_C[0], N * 2);

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