#include <stdio.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>

using namespace nvcuda;

// --- 极限几何参数 ---
#define BM 128
#define BN 128
#define BK 32   

#define WM 32
#define WN 64
#define WK 16   

__global__ void matrixMulTensorCoreFP16(half *A, half *B, float *C, int N) {
    // 1. 申请 Shared Memory 瓷砖
    __shared__ half s_A[BM][BK]; // 128 x 32 = 4096 元素
    __shared__ half s_B[BK][BN]; // 32 x 128 = 4096 元素

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * 16 + tx; // 0 到 255 

    int warpId = tid / 32;
    int warp_row = warpId / 2; // 4x2 网格
    int warp_col = warpId % 2; 

    // 2. 声明 WMMA Fragment
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag[2][4];

    // 初始化累加器
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            wmma::fill_fragment(c_frag[i][j], 0.0f);
        }
    }

    // 开始主循环
    for (int ph = 0; ph < N / BK; ++ph) {

        // 3. 🌟 一维展平安全搬运 A (256个线程，每步搬运 256 * 8 = 2048 个数)
        #pragma unroll
        for (int step = 0; step < 2; ++step) {
            // 计算当前线程在当前批次负责的“一维全局元素索引”
            int a_p_idx = step * 2048 + tid * 8; 
            
            // 将一维索引解算回 s_A 的二维行、列坐标
            int s_a_r = a_p_idx / BK;
            int s_a_c = a_p_idx % BK;

            // 映射到 Global Memory A 的真实坐标
            int g_a_row = blockIdx.y * BM + s_a_r;
            int g_a_col = ph * BK + s_a_c;

            // 强行使用 float4 从 Global 拉取并写入 Shared，数学上完美隔离、绝不踩踏
            *(float4*)&s_A[s_a_r][s_a_c] = *(float4*)&A[g_a_row * N + g_a_col];
        }

        // 4. 🌟 一维展平安全搬运 B (同样安全分两批)
        #pragma unroll
        for (int step = 0; step < 2; ++step) {
            int b_p_idx = step * 2048 + tid * 8;
            
            int s_b_r = b_p_idx / BN;
            int s_b_c = b_p_idx % BN;

            int g_b_row = ph * BK + s_b_r;
            int g_b_col = blockIdx.x * BN + s_b_c;

            *(float4*)&s_B[s_b_r][s_b_c] = *(float4*)&B[g_b_row * N + g_b_col];
        }

        // 严格同步：确保 Shared Memory 瓷砖全部就位
        __syncthreads();

        // 5. 核心计算：BK=32 内部迭代两次 WK=16
        #pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                int s_a_row_offset = warp_row * WM + i * 16;
                wmma::load_matrix_sync(a_frag, &s_A[s_a_row_offset][kk], BK);

                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    int s_b_col_offset = warp_col * WN + j * 16;
                    wmma::load_matrix_sync(b_frag, &s_B[kk][s_b_col_offset], BN);

                    // 驱动底层 Tensor Core 硬件爆发
                    wmma::mma_sync(c_frag[i][j], a_frag, b_frag, c_frag[i][j]);
                }
            }
        }

        // 同步，防止走得快的 Warp 污染下一轮瓷砖
        __syncthreads();
    }

    // 6. 倾泻结果写回 Global Memory C (FP32)
    int c_row_base = blockIdx.y * BM + warp_row * WM;
    int c_col_base = blockIdx.x * BN + warp_col * WN;

    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            float *c_store_ptr = &C[(c_row_base + i * 16) * N + (c_col_base + j * 16)];
            wmma::store_matrix_sync(c_store_ptr, c_frag[i][j], N, wmma::mem_row_major);
        }
    }
}

int main() {
    int N = 4096;
    size_t size_half = N * N * sizeof(half);
    size_t size_float = N * N * sizeof(float);

    half *h_A = (half *)malloc(size_half);
    half *h_B = (half *)malloc(size_half);
    float *h_C = (float *)malloc(size_float);

    for (int i = 0; i < N * N; ++i) {
        h_A[i] = __float2half(1.0f);
        h_B[i] = __float2half(2.0f);
    }

    half *d_A, *d_B;
    float *d_C;
    cudaMalloc(&d_A, size_half);
    cudaMalloc(&d_B, size_half);
    cudaMalloc(&d_C, size_float);

    cudaMemcpy(d_A, h_A, size_half, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_half, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(N / BN, N / BM);

    printf("🚀 终极觉醒：RTX 4060 正在全火力推进 FP16 Tensor Core 引擎...\n");

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 预热
    matrixMulTensorCoreFP16<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    matrixMulTensorCoreFP16<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    float msecTotal = 0;
    cudaEventElapsedTime(&msecTotal, start, stop);

    double flopsPerMatrixMul = 2.0 * N * N * N;
    double gflops = (flopsPerMatrixMul * 1.0e-9) / (msecTotal / 1000.0);

    cudaMemcpy(h_C, d_C, size_float, cudaMemcpyDeviceToHost);

    printf("⏱️ FP16 Tensor Core 耗时: %.2f 毫秒\n", msecTotal);
    printf("⚡ FP16 Tensor Core 性能: %.3f GFLOPS (%.3f TFLOPS)\n", gflops, gflops / 1000.0);
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