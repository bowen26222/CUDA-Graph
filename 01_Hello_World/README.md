# CUDA Hello World 程序

这是最简单的CUDA程序示例，展示了CUDA编程的基本概念。

## 程序说明

- `hello_cuda.cu`: 主程序文件
- `build.bat`: 编译脚本

## 程序功能

这个程序会：
1. 在CPU上打印 "Hello World from CPU!"
2. 显示GPU设备信息（设备名称、SM数量、内存等）
3. 在GPU上启动10个线程，每个线程打印 "Hello World from GPU!"

## 编译和运行

### Windows系统

#### 编译和运行程序
```cmd
build.bat
```

或者手动编译：
```cmd
nvcc -arch=sm_86 -o hello_cuda.exe hello_cuda.cu
```

#### 运行程序
```cmd
hello_cuda.exe
```

### Linux/Mac系统

#### 编译程序
```bash
make
```

#### 运行程序
```bash
make run
```

或者直接运行：
```bash
./hello_cuda
```

#### 清理编译文件
```bash
make clean
```

## 程序解释

1. **CUDA内核函数**: `__global__ void helloFromGPU()` 是在GPU上运行的函数
2. **内核启动**: `helloFromGPU<<<1, 10>>>()` 启动内核，参数表示1个块，每个块10个线程
3. **同步**: `cudaDeviceSynchronize()` 等待GPU完成所有任务
4. **设备信息**: 显示GPU的详细硬件信息

## 系统要求

- NVIDIA GPU
- CUDA Toolkit
- 支持CUDA的编译器

## 常见问题解决

### Windows编译问题

如果遇到 `Cannot find compiler 'cl.exe'` 错误：

1. **安装Visual Studio Build Tools**（推荐）：
   - 下载地址：https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022
   - 选择"Build tools for Visual Studio 2022"
   - 安装时选择"C++ build tools"工作负载

2. **或者安装Visual Studio Community**：
   - 下载地址：https://visualstudio.microsoft.com/vs/community/
   - 安装时选择"Desktop development with C++"工作负载

### CUDA架构警告

如果遇到架构警告，可以：
- 使用 `build.bat` 脚本，它会尝试不同的架构设置
- 或者手动指定适合您显卡的架构：
  ```cmd
  nvcc -arch=sm_75 -o hello_cuda.exe hello_cuda.cu
  ```

### 编码问题

如果遇到编码警告（如 `warning C4819`），可以：
- 使用 `build.bat` 脚本，它会自动处理编码问题
- 或者手动设置编码：
  ```cmd
  chcp 65001
  nvcc -arch=sm_86 -o hello_cuda.exe hello_cuda.cu
  ```

## 显卡架构对照表

| 显卡系列 | 架构参数 | 示例显卡 |
|---------|---------|---------|
| RTX 40系列 | sm_89 | RTX 4090, RTX 4080 |
| RTX 30系列 | sm_86 | RTX 3090, RTX 3080 |
| RTX 20系列 | sm_75 | RTX 2080, RTX 2070 |
| GTX 16系列 | sm_75 | GTX 1660, GTX 1650 |
| GTX 10系列 | sm_61 | GTX 1080, GTX 1070 | 