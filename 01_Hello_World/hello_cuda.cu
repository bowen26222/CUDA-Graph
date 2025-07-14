#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>

// CUDA内核函数 - 在GPU上运行
__global__ void helloFromGPU() {
    printf("Hello World from GPU!\n");
}

int main() {
    printf("Hello World from CPU!\n");
    
    // 获取GPU设备属性
    int dev = 0;
    cudaDeviceProp devProp;
    cudaError_t error = cudaGetDeviceProperties(&devProp, dev);
    
    if (error != cudaSuccess) {
        printf("获取设备属性时出错: %s\n", cudaGetErrorString(error));
        return -1;
    }
    
    // 打印GPU信息
    std::cout << "GPU设备 " << dev << ": " << devProp.name << std::endl;
    std::cout << "SM数量: " << devProp.multiProcessorCount << std::endl;
    std::cout << "每个线程块的共享内存大小: " << devProp.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
    std::cout << "每个线程块的最大线程数: " << devProp.maxThreadsPerBlock << std::endl;
    std::cout << "每个SM的最大线程数: " << devProp.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "每个SM的最大线程束数: " << devProp.maxThreadsPerMultiProcessor / 32 << std::endl;
    
    // 启动CUDA内核
    // 参数说明：
    // <<<1, 10>>> 表示启动1个块，每个块有10个线程
    helloFromGPU<<<1, 10>>>();
    
    // 等待GPU完成
    cudaDeviceSynchronize();
    
    return 0;
} 