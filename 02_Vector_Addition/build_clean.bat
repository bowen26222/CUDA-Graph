@echo off
nvcc -o vector_add_clean.exe vector_add_clean.cu -arch=sm_60 -O2 -std=c++14
if %errorlevel% neq 0 (
    echo Compilation failed!
    pause
    exit /b 1
)
vector_add_clean.exe
pause 