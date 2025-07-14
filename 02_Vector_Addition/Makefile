# CUDA编译器
NVCC = nvcc

# 编译选项
CFLAGS = -O2

# 目标文件
TARGET = vector_add

# 默认目标
all: $(TARGET)

# 编译规则
$(TARGET): $(TARGET).cu
	$(NVCC) $(CFLAGS) -o $(TARGET) $(TARGET).cu

# 清理规则
clean:
	rm -f $(TARGET)

# 运行规则
run: $(TARGET)
	./$(TARGET)

.PHONY: all clean run 