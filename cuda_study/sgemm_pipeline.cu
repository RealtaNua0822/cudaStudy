#include <stdio.h>
#include <cuda_runtime.h>

#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8

__global__ void matrixMulPipeline(float *A, float *B, float *C, int N) {
    // 1. 申请双倍的 Shared Memory 空间 (第二维加个 [2], 做双缓冲区)
    __shared__ float s_A[2][BK][BM]; // [Stage][8][128]
    __shared__ float s_B[2][BK][BN]; // [Stage][8][128]

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * 16 + tx; 

    float accum[TM][TN] = {0.0f};
    float reg_A[TM];
    float reg_B[TN];

    // 分工搬运坐标
    int a_load_row = tid / (BM / 4); 
    int a_load_col = (tid % (BM / 4)) * 4;
    int b_load_row = tid / (BN / 4);
    int b_load_col = (tid % (BN / 4)) * 4;

    // --- 步骤 A: 预加载阶段 (Prologue) ---
    // 在大循环开始前，把第 0 块数据强行塞进 Stage 0
    {
        int g_a_row = blockIdx.y * BM + a_load_col;
        int g_a_col = 0 * BK + a_load_row;
        #pragma unroll
        for(int i=0; i<4; ++i) {
            s_A[0][a_load_row][a_load_col + i] = A[(g_a_row + i) * N + g_a_col];
        }

        int g_b_row = 0 * BK + b_load_row;
        int g_b_col = blockIdx.x * BN + b_load_col;
        *(float4*)&s_B[0][b_load_row][b_load_col] = *(float4*)&B[g_b_row * N + g_b_col];
    }
    
    // 必须等待第 0 块完全就位
    __syncthreads();

    // --- 步骤 B: 异步双缓冲大循环 ---
    int write_stage = 1; // 接下来数据往哪里搬 (1 -> 0 -> 1 ...)
    int read_stage = 0;  // 接下来计算从哪里读 (0 -> 1 -> 0 ...)

    for (int ph = 1; ph < N / BK; ++ph) {
        
        // 1. 【发射搬运指令】把下一轮 (ph) 的数据往 write_stage 搬
        // 注意：这里硬件会发出访存请求，线程不会在这里死等数据返回，而是直接往下走！
        int g_a_row = blockIdx.y * BM + a_load_col;
        int g_a_col = ph * BK + a_load_row;
        #pragma unroll
        for(int i=0; i<4; ++i) {
            s_A[write_stage][a_load_row][a_load_col + i] = A[(g_a_row + i) * N + g_a_col];
        }

        int g_b_row = ph * BK + b_load_row;
        int g_b_col = blockIdx.x * BN + b_load_col;
        *(float4*)&s_B[write_stage][b_load_row][b_load_col] = *(float4*)&B[g_b_row * N + g_b_col];

        // 2. 【核心计算】与此同时，疯狂压榨当前 read_stage 的数据
        // 因为上一次循环末尾同步过，此时 read_stage 的数据是百分之百安全的
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                reg_A[i] = s_A[read_stage][k][ty * TM + i];
            }
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                reg_B[j] = s_B[read_stage][k][tx * TN + j];
            }
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    accum[i][j] += reg_A[i] * reg_B[j];
                }
            }
        }

        // 3. 【一箭双雕的同步】
        // 既保护了刚刚 read_stage 计算完毕，不让它被下下轮覆盖
        // 又确保了刚刚发出的 write_stage 搬运指令已经全员安全写入
        __syncthreads();

        // 4. 轮替缓冲区指针
        read_stage = write_stage;
        write_stage = 1 - write_stage;
    }

    // --- 步骤 C: 收尾阶段 (Epilogue) ---
    // 大循环退出了，说明最后一块数据已经搬完了，但最后一块数据还没算呢！
    #pragma unroll
    for (int k = 0; k < BK; ++k) {
        #pragma unroll
        for (int i = 0; i < TM; ++i) {
            reg_A[i] = s_A[read_stage][k][ty * TM + i];
        }
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            reg_B[j] = s_B[read_stage][k][tx * TN + j];
        }
        #pragma unroll
        for (int i = 0; i < TM; ++i) {
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                accum[i][j] += reg_A[i] * reg_B[j];
            }
        }
    }

    // 写回 Global Memory C
    int c_row_base = blockIdx.y * BM + ty * TM;
    int c_col_base = blockIdx.x * BN + tx * TN;
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

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(N / BN, N / BM);

    printf("🚀 RTX 4060 正在开启 Pipeline 流水线双缓冲...\n");

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 预热
    matrixMulPipeline<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    matrixMulPipeline<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    float msecTotal = 0;
    cudaEventElapsedTime(&msecTotal, start, stop);

    double flopsPerMatrixMul = 2.0 * N * N * N;
    double gflops = (flopsPerMatrixMul * 1.0e-9) / (msecTotal / 1000.0);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    printf("⏱️ Pipeline 耗时: %.2f 毫秒\n", msecTotal);
    printf("🔥 Pipeline 性能: %.3f GFLOPS\n", gflops);
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