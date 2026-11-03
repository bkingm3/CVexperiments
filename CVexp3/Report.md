## 一、实验目的
图像关键点检测及图像拼接实验报告，基于SIFT特征检测算法和FLANN特征匹配方法，实现两幅图像的特征点提取、匹配和自动拼接，掌握图像配准和融合的基本原理与实现方法。
## 二、实验原理
图像拼接依靠两张图片重叠区域的SIFT 特征点完成匹配；利用匹配点求解单应矩阵 H，把其中一张图像做透视变换对齐到另一张图像坐标系；再做坐标平移，生成足够大画布，把两张图像拷贝到画布对应位置，完成全景拼接。
## 三、实验步骤
### 步骤 1：导入依赖库
导入 OpenCV 图像处理库与 numpy 数值计算库。
### 步骤 2：读取图像并转为灰度图像
1. 使用`cv2.imread()`读取两张待拼接图片`a.jpg`、`b.jpg`；
2. 使用`cv2.cvtColor()`将彩色 BGR 图像转为灰度图。
特征提取一般在灰度图像上完成，减少计算量，彩色图仅用于最终拼接显示。
<img width="448" height="224" alt="image" src="https://github.com/user-attachments/assets/40955dea-f463-421c-8c29-00fa4ccbe8ea" />
### 步骤 3：SIFT 关键点检测与特征描述子提取
1. 创建 SIFT 检测器对象`sift = cv2.SIFT_create()`；
2. 调用`detectAndCompute()`，分别对灰度图 a、灰度图 b 提取关键点 kp和特征描述子 des
- 关键点：图像上具有旋转、尺度不变性的兴趣点；
- 描述子：关键点周边局部特征向量，用于后续特征匹配。
<img width="342" height="62" alt="image" src="https://github.com/user-attachments/assets/870cc334-8a43-436c-90a3-772014f74647" />
### 步骤 4：FLANN 匹配器进行 k‑近邻特征匹配
1. 设置 FLANN 索引参数（KD‑Tree）与搜索参数，构建 FLANN 匹配器；
2. 使用`knnMatch`设置`k=2`，对两组描述子做匹配，每个特征点返回 2 个最相近匹配结果。
FLANN 是快速近似最近邻匹配，相比暴力匹配速度更快，适合特征点数量多的场景。
<img width="378" height="99" alt="image" src="https://github.com/user-attachments/assets/9a7a0284-ae19-4b1e-8ae6-c622217be4fd" />
### 步骤 5：Lowe 比率测试筛选优质匹配对
遍历 k=2 得到的匹配对`m,n`，按照 Lowe 规则：如果最佳匹配距离 m.distance 小于次优匹配距离 n.distance 的 0.7 倍，则认为是可靠匹配，存入`good_matches`。
该步骤剔除模糊、错误匹配，阈值一般取 0.7~0.8。
<img width="215" height="77" alt="image" src="https://github.com/user-attachments/assets/db9878e9-6d59-4de8-b1fa-fafc61233c72" />
### 步骤 6：绘制匹配关键点可视化匹配结果
调用`cv2.drawMatches()`绘制两张图像以及筛选后的匹配连线，不绘制孤立关键点，得到匹配可视化图，便于观察匹配效果。
### 步骤 7：提取匹配点坐标，RANSAC 求解单应矩阵 H
1. 从`good_matches`中分别取出图像 b、图像 a 上匹配关键点的像素坐标，整理成`np.float32`格式；
2. 使用`cv2.findHomography()`，采用 RANSAC 随机抽样一致性算法，求解从图像 b 映射到图像 a 的**单应矩阵 H**，同时剔除外点（错误匹配点），RANSAC 重投影误差阈值设置 5.0。
单应矩阵描述两张图像平面之间的透视变换关系。
### 步骤 8：计算变换后图像四角，确定全景画布大小
1. 获取原图 a、原图 b 的高、宽；
2. 取图像 b 四个角点，利用`cv2.perspectiveTransform()`结合 H 矩阵，算出 b 图经过透视变换之后的 4 个角点；
3. 将 a 图原始四角坐标和变换后 b 图四角坐标合并，求出全部角点的最小、最大 x、y 坐标，确定全景图完整边界
<img width="526" height="479" alt="image" src="https://github.com/user-attachments/assets/94f75252-6c45-4a4d-adc4-80d960c88569" />
### 步骤 9：构造平移矩阵，完成透视变换
1. 图像 b 透视变换后会出现负坐标，需要构造平移矩阵，把整体图像向右、向下平移，保证全部像素坐标大于 0；
2. 将平移矩阵与单应矩阵做矩阵乘法，得到总的变换矩阵；
3. 使用`cv2.warpPerspective()`把图像 b 变换到全景画布上，画布尺寸为`(x_max‑x_min, y_max‑y_min)`
### 步骤 10：把图像 a 拷贝到全景画布对应位置
利用切片操作，将原始图像 a 复制到大画布正确位置，完成两张图像合并。
<img width="561" height="175" alt="image" src="https://github.com/user-attachments/assets/5465ffbf-2b7b-4096-afc7-283a08522e6c" />
### 步骤 11：图像显示、等待按键、资源释放
1. 显示匹配关键点效果图、拼接全景结果图；
2. `cv2.waitKey(0)`阻塞等待键盘按键，任意按键后关闭所有 OpenCV 窗口。
可选：取消注释`cv2.imwrite`，将拼接结果保存为图片文件。
<img width="373" height="184" alt="image" src="https://github.com/user-attachments/assets/3c9744b2-1c19-4f47-baf2-81f1f43c1a11" />
### 实验现象与结果
1. 弹出第一个窗口：显示两张图片以及 SIFT 匹配点连线，可以观察匹配点分布；
2. 弹出第二个窗口：输出两张图片拼接后的全景图像；
3. 按下任意按键，关闭全部窗口，若开启保存代码，会生成拼接完成的全景图片。





































