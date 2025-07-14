#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>

// =============================
// CUDA 向量加法核函数
// 每个线程计算一个元素的加法
// =============================
__global__ void vectorAdd(float* a, float* b, float* c, int n) {
    // 计算全局线程索引
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // 边界检查，防止越界访问
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    // ========== 参数设置 ==========
    int n = 1000000; // 向量元素个数
    size_t size = n * sizeof(float); // 内存字节数

    printf("=== CUDA Vector Addition Demo ===\n\n");
    printf("Array size: %d elements\n", n);
    printf("Memory size: %.2f MB\n\n", size / (1024.0 * 1024.0));

    // ========== 统一内存分配 ==========
    float *a, *b, *c;
    cudaMallocManaged(&a, size); // 分配a
    cudaMallocManaged(&b, size); // 分配b
    cudaMallocManaged(&c, size); // 分配c

    // ========== 初始化数据 ==========
    printf("Initializing arrays...\n");
    for (int i = 0; i < n; i++) {
        a[i] = (float)i;        // a: 0, 1, 2, ...
        b[i] = (float)i * 2;    // b: 0, 2, 4, ...
    }

    // ========== 配置线程块和网格 ==========
    int threadsPerBlock = 256; // 每个块的线程数
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock; // 块数
    printf("Launching kernel with %d blocks, %d threads per block\n", blocksPerGrid, threadsPerBlock);
    printf("Total threads: %d\n\n", blocksPerGrid * threadsPerBlock);

    // ========== 创建CUDA事件用于计时 ==========
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // ========== 启动计时 ==========
    cudaEventRecord(start, 0);

    // ========== 启动核函数 ==========
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(a, b, c, n);

    // ========== 等待核函数完成并停止计时 ==========
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // ========== 计算核函数执行时间 ==========
    float kernelTime;
    cudaEventElapsedTime(&kernelTime, start, stop);
    printf("GPU kernel execution time: %.3f ms\n", kernelTime);

    // ========== 检查核函数启动错误 ==========
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(error));
        return -1;
    }

    // ========== 校验结果（前10个元素） ==========
    printf("\nVerifying results...\n");
    printf("First 10 elements:\n");
    for (int i = 0; i < 10; i++) {
        printf("a[%d] = %.1f, b[%d] = %.1f, c[%d] = %.1f\n", i, a[i], i, b[i], i, c[i]);
    }

    // ========== 检查所有结果是否正确 ==========
    bool correct = true;
    for (int i = 0; i < n; i++) {
        if (c[i] != a[i] + b[i]) {
            correct = false;
            break;
        }
    }
    if (correct) {
        printf("\nAll results are correct!\n");
    } else {
        printf("\nSome results are incorrect!\n");
    }

    // ========== 计算内存带宽 ==========
    float bandwidth = (size * 3) / (kernelTime / 1000.0) / (1024.0 * 1024.0 * 1024.0); // GB/s
    printf("Memory bandwidth: %.2f GB/s\n", bandwidth);

    // ========== 释放CUDA事件 ==========
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // ========== 释放内存 ==========
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    printf("\nMemory freed successfully.\n");
    return 0;
} 