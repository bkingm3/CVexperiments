# 实验四：MLP-多层感知机

## 一、实验目的

本次实验围绕多层感知机 MLP 展开，借助 PyTorch 搭建神经网络完成手写数字图像分类任务。通过实践搭建、训练、调优 MLP 模型，体会神经网络的基础构成要素，理解前向传播、反向传播的完整流程，熟悉全连接网络在图像分类任务上的使用方式，同时掌握网络结构调整、正则化、优化器选型等模型调优手段。

## 二、实验内容

### 1. 导入依赖库

实验需要导入深度学习框架、数据处理以及可视化相关模块。

- `torch`：PyTorch 核心库，负责张量运算，是整个深度学习实验的基础；
- `torch.nn`：神经网络模块，封装网络层、模型基类、损失相关组件；
- `torch.nn.functional`：函数式接口，提供激活函数、损失函数等功能；
- `torch.optim`：优化器工具库，包含 SGD、Adam 等主流参数更新算法；
- `numpy`：数值计算工具，用于数组运算；
- `matplotlib.pyplot`：绘图工具，实现训练曲线、混淆矩阵等结果可视化；
- `torchvision`：计算机视觉工具包，负责 MNIST 数据集读取、图像预处理变换。

<img width="348" height="136" alt="image" src="https://github.com/user-attachments/assets/b53fd28c-0983-4804-a2ba-d9c9cc637162" />

整套库分工明确：`torch.nn`用来搭建网络模型；`torchvision`完成数据集加载与图像预处理；`torch.optim`实现模型参数迭代更新；`torch.nn.functional`计算训练损失；`matplotlib`用于直观观察训练效果；`numpy`承担辅助数值计算工作，共同构成完整深度学习实验流程。

### 2. 硬件设备检测

自动判断是否存在 CUDA 显卡，优先使用 GPU 运算，无 GPU 则使用 CPU，打印设备信息便于调试。

<img width="459" height="96" alt="image" src="https://github.com/user-attachments/assets/f12bd7e2-4534-44ef-842a-d6b5ee44e539" />

### 3. 加载 MNIST 数据集

设置`batch_size=2048`，加载 MNIST 手写数字数据集。训练集开启 shuffle 打乱样本，测试集不打乱；使用`transforms.ToTensor()`将图片转为张量。通过`DataLoader`构建批次迭代加载器。

<img width="406" height="271" alt="image" src="https://github.com/user-attachments/assets/739450e7-3837-47a1-bf0d-26ae194dc58c" />

> MNIST 数据集包含 60000 张 28×28 训练图像、10000 张 28×28 测试图像；MLP 接收一维向量，训练时需要将`[batch,1,28,28]`图像展平为`[batch,784]`。

### 4. 定义基础 MLP 模型、优化器

自定义`mlp`类继承`nn.Module`；基础网络两层全连接：`784→128→10`，隐藏层使用 ReLU 激活。
实例化模型并迁移至对应设备，优化器选用 SGD，学习率 lr=0.01，训练总轮数`epochs=10`。


<img width="392" height="638" alt="image" src="https://github.com/user-attachments/assets/a33279ea-2c10-4d45-8121-0686c2e8559e" />

<img width="200" height="151" alt="image" src="https://github.com/user-attachments/assets/cfdf7605-2b6b-4739-b237-3d370a076c09" />


**训练‑测试循环流程说明**

1. `model.train()`开启训练模式；遍历训练集，`x.view()`把图像展平为一维向量；
2. `optimizer.zero_grad()`清空梯度；前向传播得到输出；`cross_entropy`计算分类损失；
3. `loss.backward()`反向传播求梯度；`optimizer.step()`更新权重；
4. 每轮训练结束后切换`model.eval()`，`torch.no_grad()`关闭梯度计算，在测试集上计算损失和准确率；
5. 将每轮测试 loss、acc 存入列表，打印日志，循环结束后绘制测试损失、测试准确率变化曲线。

### 5. 模型改进

基础两层网络表达能力有限，且 SGD 学习率 0.01 收敛速度一般。对模型做如下改进：

1. 加深网络，结构：`784 → 256 →128 →64 →10`；
2. 激活函数替换为 GELU，梯度更加平滑；
3. 每一层全连接后增加`BatchNorm1d`批归一化，缓解梯度消失；
4. 添加`Dropout(0.3)`做正则化，抑制过拟合；
5. 优化器更换为 Adam，学习率设置`lr=1e‑3`；
6. 沿用同样的训练‑测试循环，保存测试损失、准确率

<img width="345" height="527" alt="image" src="https://github.com/user-attachments/assets/bd292f85-0cf0-40a5-9772-06112c6e2405" />

<img width="380" height="230" alt="image" src="https://github.com/user-attachments/assets/6e5a78b0-97c8-4670-b1a6-2a1aa649bdff" />

<img width="215" height="292" alt="image" src="https://github.com/user-attachments/assets/3d949b2a-4f61-4f69-8158-21919a40182b" />

## 三、实验结果与分析

### 1. 基础模型结果

基础模型网络结构：`784‑128‑10`，ReLU 激活，SGD 优化器`lr=0.01`，训练轮数 10 轮。
从打印日志与绘制的曲线可以观察：测试损失随着 epoch 增加逐步下降，测试准确率持续上升，模型可以正常收敛。
由于网络较浅、学习率偏小，收敛速度比较平缓；最终可以达到一个不错的基础分类效果，但仍然存在性能上限。测试集损失曲线后期下降趋于平缓，准确率提升幅度变小，反映模型表达能力受限。

### 2. 改进模型结果

改进后网络为多层结构，叠加 BatchNorm、Dropout，GELU 激活，Adam 优化器。
对比基础模型：

1. 损失下降速度明显更快，前期 epoch 损失就可以降到很低水平；
2. 最终测试准确率显著高于基础版本；
3. BatchNorm 稳定特征分布，缓解深层网络梯度消失；Dropout 起到正则化，降低过拟合风险；GELU 平滑梯度、Adam 自适应学习率进一步提升训练效率。

从曲线可以看出，改进模型的测试损失更低，测试准确率更高，泛化性能得到明显提升。

### 3. 结论

仅仅依靠增加训练轮数、调整基础学习率，浅层 MLP 性能提升有限。通过加深网络、选用更合适的激活函数、引入批归一化与 Dropout 正则化，搭配自适应优化器，能够有效提升模型的拟合能力、收敛速度与泛化能力。

## 四、实验总结与心得

本次实验使用 Jupyter 分段代码完成 MLP 手写数字分类，完整复现了数据集加载、模型定义、训练循环、测试评估、结果可视化整套流程。
通过实验我体会到：

1. 训练轮数、学习率会直接影响 SGD 的收敛效果；学习率过小会导致收敛缓慢；
2. 在训练循环中每轮结束后在测试集上评估，并且保存指标绘图，可以直观观察模型是否收敛、是否出现过拟合；
3. 加深网络不能只简单堆叠全连接层，需要配合批归一化、正则化策略，否则容易出现训练不稳定或者过拟合；
4. Adam 自适应优化器、GELU 激活函数相比 SGD、ReLU 更适合深层网络；Dropout 与 BatchNorm 组合可以兼顾训练稳定性与泛化能力。























