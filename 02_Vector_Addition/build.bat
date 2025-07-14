@echo off
nvcc -o vector_add.exe vector_add.cu -arch=sm_60 -O2 -std=c++14
if %errorlevel% neq 0 (
    echo 编译失败，请检查CUDA和VS环境！
    pause
    exit /b 1
)
vector_add.exe
pause 