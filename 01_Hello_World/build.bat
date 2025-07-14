@echo off
echo 正在编译CUDA Hello World程序...

REM 设置代码页为UTF-8
chcp 65001 >nul

REM 尝试现代架构
echo 尝试现代架构...
nvcc -arch=sm_86 -o hello_cuda.exe hello_cuda.cu
if %ERRORLEVEL% EQU 0 goto success

REM 尝试较新架构
echo 尝试较新架构...
nvcc -arch=sm_75 -o hello_cuda.exe hello_cuda.cu
if %ERRORLEVEL% EQU 0 goto success

REM 尝试通用架构
echo 尝试通用架构...
nvcc -arch=sm_61 -o hello_cuda.exe hello_cuda.cu
if %ERRORLEVEL% EQU 0 goto success

REM 尝试抑制警告
echo 尝试抑制警告...
nvcc -Wno-deprecated-gpu-targets -o hello_cuda.exe hello_cuda.cu
if %ERRORLEVEL% EQU 0 goto success

echo 所有编译方法都失败了！
echo.
echo 请检查：
echo 1. CUDA Toolkit安装
echo 2. Visual Studio Build Tools安装
echo 3. GPU兼容性
goto end

:success
echo 编译成功！
echo 正在运行程序...
echo.
hello_cuda.exe

:end
pause 