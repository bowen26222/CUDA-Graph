# CUDA 向量加法演示

这是一个简单的CUDA程序，演示如何在GPU上进行向量加法运算。

## 程序功能

- 在GPU上并行计算两个大数组的加法
- 使用CUDA事件进行精确计时
- 验证计算结果的正确性
- 计算内存带宽性能指标

## 文件说明

- `vector_add.cu` - 主程序文件（英文注释版本，避免编码问题）
- `build_utf8.bat` - 支持UTF-8编译脚本
- `build.bat` - 标准编译脚本
- `Makefile` - Linux/Mac编译脚本

## 编译方法

### Windows 用户

**推荐使用UTF-8编译脚本：**
```bash
build_utf8.bat
```

**或者使用标准编译脚本：**
```bash
build.bat
```

### Linux/Mac 用户

```bash
make
```

## 编码问题解决方案

如果遇到编码警告（warning C4819），程序已经做了以下处理：

1. **代码注释改为英文** - 避免中文字符编码问题
2. **UTF-8编译脚本** - 使用 `/utf-8` 和 `/wd4819` 参数
3. **多重编译尝试** - 如果UTF-8编译失败，自动尝试标准编译

## 程序输出示例

```
=== CUDA Vector Addition Demo ===

Array size: 1000000 elements
Memory size: 12.00 MB

Initializing arrays...
Launching kernel with 3907 blocks, 256 threads per block
Total threads: 1000192

GPU kernel execution time: 0.123 ms

Verifying results...
First 10 elements:
a[0] = 0.0, b[0] = 0.0, c[0] = 0.0
a[1] = 1.0, b[1] = 2.0, c[1] = 3.0
...

✅ All results are correct!
Memory bandwidth: 292.68 GB/s

Memory freed successfully.
```

## 性能特点

- **并行计算** - 使用100万个线程同时计算
- **统一内存** - 自动管理CPU和GPU内存
- **精确计时** - 使用CUDA事件测量GPU执行时间
- **结果验证** - 确保计算结果的正确性

## 技术细节

- **线程配置**: 256线程/块，3907个块
- **内存管理**: 使用`cudaMallocManaged`统一内存
- **计时方法**: CUDA事件计时，精度可达微秒级
- **错误处理**: 检查内核启动错误和内存分配错误

## 故障排除

### 编译错误
1. 确保安装了CUDA Toolkit
2. 确保安装了Visual Studio Build Tools
3. 检查GPU兼容性

### 编码问题
- 使用`build_utf8.bat`脚本
- 程序已使用英文注释避免编码问题

### 性能问题
- 调整`threadsPerBlock`参数
- 检查GPU架构设置
- 监控GPU温度和内存使用