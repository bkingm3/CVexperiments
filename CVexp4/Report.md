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
实例化模型并迁移至对应设备，优化器选用 SGD，**学习率 lr=0.01**，训练总轮数`epochs=10`。
同时定义两个列表`test_loss_list`、`test_acc_list`，用来保存每一轮测试集损失与准确率，用于后续绘图。

































