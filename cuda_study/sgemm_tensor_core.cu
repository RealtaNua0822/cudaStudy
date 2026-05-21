#include <stdio.h>
#include <cuda_runtime.h>
// 1. 引入标准的 mma 头文件
#include <mma.h>

// 2. 正确的命名空间
using namespace nvcuda;

// 宏定义保持不变
#define BM 64
#define BN 128
#define BK 16

#define WM 32
#define WN 32
#define WK 16

__global__ void matrixMulTensorCore(float *A, float *B, float *C, int N) {
    // 申请用于搬运的 Shared Memory 瓷砖
    __shared__ float s_A[BM][BK]; // 64 x 16
    __shared__ float s_B[BK][BN]; // 16 x 128

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * 16 + tx; 

    int warpId = tid / 32;
    int warp_row = warpId / 4; 
    int warp_col = warpId % 4; 

    // 3. 切换为硬核硬件原生支持的 tf32 模式（输入截断，FP32累加）
    wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 8, float> c_frag[2][2];

    // 初始化累加器片段为 0
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            wmma::fill_fragment(c_frag[i][j], 0.0f);
        }
    }
    
    // ... 后面的搬运、循环、load_matrix_sync、mma_sync、store_matrix_sync 完全保持原样即可 ...

    // 4. 计算当前线程搬运 Global -> Shared 的分工坐标
    int a_load_row = tid / (BM / 4); // 256个线程搬运 16*64=1024 个数
    int a_load_col = (tid % (BM / 4)) * 4;
    int b_load_row = tid / (BN / 8); // 256个线程搬运 16*128=2048 个数
    int b_load_col = (tid % (BN / 8)) * 8;

    // 开始主循环
    for (int ph = 0; ph < N / BK; ++ph) {

        // 5. 强行拉满 float4 向量化总线进行搬运
        int g_a_row = blockIdx.y * BM + a_load_col;
        int g_a_col = ph * BK + a_load_row;
        #pragma unroll
        for(int i=0; i<4; ++i) {
            s_A[a_load_col + i][a_load_row] = A[(g_a_row + i) * N + g_a_col];
        }

        int g_b_row = ph * BK + b_load_row;
        int g_b_col = blockIdx.x * BN + b_load_col;
        *(float4*)&s_B[b_load_row][b_load_col] = *(float4*)&B[g_b_row * N + g_b_col];
        *(float4*)&s_B[b_load_row][b_load_col + 4] = *(float4*)&B[g_b_row * N + g_b_col + 4];

        // 严格同步，确保全员瓷砖就位
        __syncthreads();

        // 6. 核心战斗：调用 Tensor Core 硬件
        // 每个 Warp 负责 32x32 的大块，内部拆成 2x2 个 16x16 的小硬块
        #pragma unroll
        for (int kk = 0; kk < BK; kk += 8) {
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                // 加载 A 的片段到寄存器：从 Shared Memory 载入
                int s_a_row_offset = warp_row * WM + i * 16;
                int s_a_col_offset = kk;
                wmma::load_matrix_sync(a_frag, &s_A[s_a_row_offset][s_a_col_offset], BK);

                #pragma unroll
                for (int j = 0; j < 2; ++j) {
                    // 加载 B 的片段
                    int s_b_row_offset = kk;
                    int s_b_col_offset = warp_col * WN + j * 16;
                    wmma::load_matrix_sync(b_frag, &s_B[s_b_row_offset][s_b_col_offset], BN);

                    // 🌟 核心硬核指令：让硬件 Tensor Core 喷射算力
                    wmma::mma_sync(c_frag[i][j], a_frag, b_frag, c_frag[i][j]);
                }
            }
        }

        // 同步，防止走得快的 Warp 污染下一轮的 Shared Memory
        __syncthreads();
    }

    // 7. 倾泻结果：将 4 个 c_frag 寄存器片段合并写回 Global Memory C
    int c_row_base = blockIdx.y * BM + warp_row * WM;
    int c_col_base = blockIdx.x * BN + warp_col * WN;

    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            float *c_store_ptr = &C[(c_row_base + i * 16) * N + (c_col_base + j * 16)];
            // 使用 wmma::store_matrix_sync 直接将寄存器片段倾泻到内存，指定主序为行主序
            wmma::store_matrix_sync(c_store_ptr, c_frag[i][j], N, wmma::mem_row_major);
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

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(N / BN, N / BM);

    printf("🔥 RTX 4060 正在唤醒硬核 Tensor Core 矩阵乘法引擎...\n");

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 预热
    matrixMulTensorCore<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    matrixMulTensorCore<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    float msecTotal = 0;
    cudaEventElapsedTime(&msecTotal, start, stop);

    double flopsPerMatrixMul = 2.0 * N * N * N;
    double gflops = (flopsPerMatrixMul * 1.0e-9) / (msecTotal / 1000.0);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    printf("⏱️ Tensor Core 耗时: %.2f 毫秒\n", msecTotal);
    printf("🚀 Tensor Core 性能: %.3f GFLOPS\n", gflops);
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