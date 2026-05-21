# CUDA 性能测试记录

## 环境
- 机器：`realtanua-Legion`
- GPU：`RTX 4060`
- 测试目录：`~/ai-infra/cuda_study`
- 可执行文件：`sgemm_naive`

## 1) 编译并直接运行

```bash
nvcc -O3 sgemm_naive.cu -o sgemm_naive
./sgemm_naive
```

输出：

```text
🚀 RTX 4060 开始疯狂压榨 4096 Naive 算子...
🎉 计算完成！
⏱️ 真实耗时: 259.142 毫秒
📊 实际性能: 530.362 GFLOPS
🔍 验证坐标[0][0]结果: 8192 (预期值: 8192)
```

## 2) 使用 Nsight Compute (`ncu`) 分析

```bash
sudo /usr/local/cuda-13.2/bin/ncu --section SpeedOfLight ./sgemm_naive
```

关键输出（摘录）：

```text
==PROF== Profiling "matrixMulNaive" - 0: 0%....50%....100% - 7 passes
⏱️ 真实耗时: 1838.59 毫秒
📊 实际性能: 74.7525 GFLOPS

[72630] sgemm_naive@127.0.0.1
  matrixMulNaive(float *, float *, float *, int) (128, 128, 1)x(32, 32, 1), Context 1, Stream 7, Device 0, CC 8.9

GPU Speed Of Light Throughput:
- DRAM Frequency: 7.99 Ghz
- SM Frequency: 1.89 Ghz
- Elapsed Cycles: 385,960,757
- Memory Throughput: 92.77%
- DRAM Throughput: 19.59%
- Duration: 204.21 ms
- L1/TEX Cache Throughput: 92.86%
- L2 Cache Throughput: 5.90%
- Compute (SM) Throughput: 92.77%

INF: This workload is utilizing greater than 80.0% of the available compute or memory performance.
```

## 3) 结果备注
- 程序计算正确性验证通过（`C[0][0] = 8192`）。
- `ncu` 采样会显著拉长总耗时（多次 pass + profiler 开销），因此与直接运行耗时不可直接对比。
- 从 SpeedOfLight 指标看，当前 Naive Kernel 在该测试中已经体现出较高的单元利用率，但整体 GFLOPS 与理论峰值仍有差距，后续可尝试 shared memory tiling / vectorized load / Tensor Core 路径优化。
