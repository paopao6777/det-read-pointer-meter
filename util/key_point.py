import torch
import numpy as np
import cv2
from skimage import morphology

# 刻度线中心点搜索
def line_point_search(self, x):
    '''
    对output的每个特定通道（指针、刻度盘、文本）应用Sigmoid函数。Sigmoid函数的作用是将模型输出的原始分数转换为介于0和1之间的概率值，表示某个像素点属于目标类别的概率。
    提取特定通道的预测：
    output[0, 0, :, :]：从模型输出中提取第一个样本的第一个通道（假设为指针）的所有像素值。
    output[0, 1, :, :]：提取第一个样本的第二个通道（假设为刻度盘）的所有像素值。

    二值化处理：
    对Sigmoid处理后的结果进行二值化，以便将每个像素分类为目标类别或背景。这里使用不同的阈值对不同的特征进行二值化：指针和刻度盘使用0.5作为阈值，而文本使用0.7作为阈值。这意味着，只有当模型对某个像素点属于特定类别的置信度超过这个阈值时，该像素点才会被分类为该类别。
    .astype(np.uint8)：将二值化后的布尔数组转换为uint8类型，通常用于表示图像数据。这里的转换使得二值化的结果可以作为二值图像进行处理和可视化。
    结果解释：

    pointer_pred、dail_pred分别表示处理后的指针、刻度的二值化预测图像。这些图像中，值为1的像素点代表模型预测该位置为相应的类别，而值为0的像素点则代表背景或其他类别。
    '''

    # 对预测结果进行Sigmoid激活处理,x为分割部分的输出
    pointer_pred = torch.sigmoid(x[0, 0, :, :]).data.cpu().numpy()  # 指针通道的预测
    dail_pred = torch.sigmoid(x[0, 1, :, :]).data.cpu().numpy()  # 刻度盘通道的预测

    # 将Sigmoid处理后的结果进行二值化：指针和刻度线使用阈值0.5，文本使用0.7
    pointer_pred = (pointer_pred > 0.5).astype(np.uint8)  # 二值化指针图像
    dail_pred = (dail_pred > 0.5).astype(np.uint8)  # 二值化刻度盘图像

    '''
    这段代码调用了filter方法来处理 dail_pred 的二值化预测图像，
    目的是过滤掉一些可能的噪声或不符合条件的连通区域，从而改善最终的检测结果。
    filter方法应用于图像上，以去除小于特定尺寸（由参数n确定）的连通组件（即图像中相连的像素块）。
    
    对dail_pred的处理：dail_label=self.filter(dail_pred, n=30)这行代码将dail_pred（刻度线的二值化预测图像）作为输入，并设置参数n=30。
    这意味着，在dail_pred图像中，所有像素数量少于30的连通区域都将被过滤掉，只保留较大的连通区域。
    '''
    # 过滤掉不符合条件的小区域
    dail_label = self.filter(dail_pred, n=30)  # 对刻度盘的预测结果进行过滤，去掉小的区域

    # cv2.imshow("srtc",text_pred*255)
    # cv2.imshow("1", pointer_pred * 255)
    # cv2.waitKey(0)

    # 使用OpenCV的findContours函数提取刻度盘区域的轮廓
    dail_edges = dail_label * 255  # 处理后的二值化图像转换为255值
    dail_edges = dail_edges.astype(np.uint8)  # 转换为uint8类型，以便findContours使用
    dail_contours, _ = cv2.findContours(dail_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 查找刻度盘轮廓

    # 提取刻度盘区域的中心点
    std_point = []  # 用于存储刻度线区域的中心点
    for i in range(len(dail_contours)):
        rect = cv2.minAreaRect(dail_contours[i])  # 获取最小外接矩形
        std_point.append((int(rect[0][0]), int(rect[0][1])))  # rect[0][0] 是矩形的中心点的 x 坐标，rect[0][1] 是矩形的中心点的 y 坐标。

    # 如果没有检测到刻度盘，返回默认值
    if len(std_point) <= 2:
        return pointer_pred, dail_label, (None, None), None

    # 返回指针预测图、刻度线图像和中心点
    return pointer_pred, dail_label, std_point

# 确定旋转中心o1
def circle_point(self, std_point):
    '''
    通过 std_point 中的点来寻找它们所在的圆周的圆心位置。
    假设 std_point 是一些落在同一个圆周上的点，尝试拟合圆形并计算圆心。
    '''
    # 如果点数太少，无法进行拟合，直接返回None
    if len(std_point) < 3:
        return None

    # 将 std_point 转换为 numpy 数组，方便后续计算
    points = np.array(std_point, dtype=np.float32)

    # 使用 cv2 提供的拟合方法拟合一个最小外接圆 o1=center坐标, r=radius半径
    o1, r = cv2.minEnclosingCircle(points)

    # cv2.minEnclosingCircle 返回的 center 是一个元组 (x, y)，radius 是半径
    # center 即为拟合圆的圆心坐标
    o1 = (int(o1[0]), int(o1[1]))  # 转换为整数坐标

    # 输出拟合圆心和半径（可以用于调试或可视化）
    print("拟合圆心位置：", o1)
    print("拟合圆的半径：", r)

    return o1, r

# 确定指针骨架、指针尖端坐标
def pointer_skeleton(self, ori_img, pointer_pred, o1):
    # 实施骨架算法，细化指针掩膜
    pointer_skeleton = morphology.skeletonize(pointer_pred)  # 使用细化算法对指针掩膜进行骨架提取
    pointer_edges = pointer_skeleton * 255  # 将骨架图像的值放大到0或255（适应OpenCV的图像格式）
    pointer_edges = pointer_edges.astype(np.uint8)  # 转换为uint8类型，以便用于OpenCV的操作

    # 使用霍夫变换从指针骨架中提取线条
    pointer_lines = cv2.HoughLinesP(pointer_edges, 1, np.pi / 180, 10, np.array([]), minLineLength=10, maxLineGap=400)
    # 霍夫变换参数：1：距离分辨率，单位像素 np.pi / 180：角度分辨率（弧度）10：累加器阈值，越高越严格 minLineLength=10：检测到的最小线段长度 maxLineGap=400：线段之间的最大间隙，如果小于这个间隙，则认为是同一条线

    u1, u2 = None, None  # 初始化两个变量，用于保存指针线段的两个端点坐标

    try:
        # 选择霍夫变换检测到的第一条线段
        for x1, y1, x2, y2 in pointer_lines[0]:
            u1 = (x1, y1)  # 指针线段的起点坐标
            u2 = (x2, y2)  # 指针线段的终点坐标
            # 在图像上绘制指针线条，颜色为 (118, 198, 165)，线宽为 2
            cv2.line(ori_img, (x1, y1), (x2, y2), (118, 198, 165), 2)
    except TypeError:
        # 如果未能检测到指针线条，返回错误信息
        return "can not detect pointer"

    # 计算 u1 和 u2 到圆心 o1 的欧氏距离
    def euclidean_distance(p1, p2):
        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    distance_u1 = euclidean_distance(u1, o1)  # 计算 u1 到圆心 o1 的距离
    distance_u2 = euclidean_distance(u2, o1)  # 计算 u2 到圆心 o1 的距离

    # 判断哪个点更远
    if distance_u1 > distance_u2:
        U = u1  # u1 更远，u1 是 U
    else:
        U = u2  # u2 更远，u2 是 U

    # 返回选中的 U 点
    return U