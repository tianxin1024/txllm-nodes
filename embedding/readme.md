# 构建Zhilight项目, 测试单元

## 1. 步骤
* 1. install pybind11
```bash
git clone https://github.com/pybind/pybind11.git
```
* 2. 添加bmengine第三方库


## 2. 运行
```bash
make
```

## 3. 调试
```bash
gdb python3
set directory ..
run ../test_embedding.py
break layer_embedding.cpp:xxx
run
layout src
```

## 4. 测试
${\lvert \text{input} - \text{other} \rvert \leq \texttt{atol} + \texttt{rtol} \times \lvert \text{other} \rvert}$
* input：第一个要比较的张量。
* other：第二个要比较的张量。
* rtol：相对容忍度，默认值为 1e-05。
* atol：绝对容忍度，默认值为 1e-08。
* equal_nan：如果为 True，则会将两个 NaN 值视为相等，默认值为 False。   

## 参考
[ZhiLight](https://github.com/zhihu/ZhiLight)
