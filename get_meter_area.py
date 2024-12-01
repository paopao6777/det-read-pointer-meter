import torch
import numpy as np
from yolov8.ultralytics.nn.tasks import attempt_load_weights  # 尝试加载YOLOv5模型
from yolov8.ultralytics.data.augment import LetterBox
from yolov8.ultralytics.utils.ops import non_max_suppression, scale_coords
from yolov8.ultralytics.utils.torch_utils import select_device

import cv2
from random import randint
import os
import time


class Detector(object):
    def __init__(self):
        self.img_size = 640  # 输入图像尺寸
        self.threshold = 0.6  # 检测阈值
        self.max_frame = 160  # 最大帧数（未在此代码中使用）
        self.init_model()  # 初始化模型

    def init_model(self):
        self.weights = 'yolov8/best.pt'  # 模型权重文件路径
        # 如果GPU可用则使用GPU，否则使用CPU
        self.device = '0' if torch.cuda.is_available() else 'cpu'
        self.device = select_device(self.device)
        ensemble = attempt_load_weights(self.weights, device=self.device)  # 加载模型
        ensemble.to(self.device).eval()  # 将模型设置为评估模式
        ensemble.half()  # 将模型转为半精度以加速计算
        self.m = ensemble
        # 获取模型类别名称
        self.names = ensemble.module.names if hasattr(ensemble, 'module') else ensemble.names

        # 输出模型类型和类别名称
        # print("Model type:", type(self.m))
        # print("Model loaded successfully.")
        # print("Class names:", self.names)
        # print("Number of classes:", len(self.names))

        # 创建名称到ID的映射字典
        self.name_to_id = {name: cls_id for cls_id, name in self.names.items()}

        # 为每个类别随机生成颜色
        self.colors = [
            (randint(0, 255), randint(0, 255), randint(0, 255)) for _ in self.names
        ]

    def preprocess(self, img):
        # 对图像进行预处理以适应模型输入
        img0 = img.copy()  # 复制原图像
        # cv2.imshow("original_image.jpg", img0)  # 保存原始图像
        letterbox = LetterBox(new_shape=(self.img_size, self.img_size))
        # 调整图像大小并保持比例
        img = letterbox(image=img)  # 调用 __call__ 方法
        # cv2.imshow("resized_image.jpg", img)  # 保存调整后的图像
        # BGR转RGB，HWC转CHW, High x Width x Color[:, :, ::-1]即前两个元素不变，第三个颠倒顺序BGR转RGB。
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half()  # 转为半精度
        img /= 255.0  # 归一化图像
        if img.ndimension() == 3:
            img = img.unsqueeze(0)  # 为图像添加批次维度

        return img0, img

    def plot_bboxes(self, image, bboxes, line_thickness=None):
        # 在图像上绘制边界框和类别标签

        # 计算线条/字体厚度，如果未指定，则根据图像尺寸动态计算
        tl = line_thickness or round(
            0.002 * (image.shape[0] + image.shape[1]) / 2) + 1  # 动态计算线条厚度

        # 遍历所有边界框
        for (x1, y1, x2, y2, cls_name, conf) in bboxes:
            # 根据类别名称获取类别ID和颜色
            cls_id = self.name_to_id[cls_name]
            color = self.colors[cls_id]
            # 定义边界框的左上角和右下角坐标
            c1, c2 = (x1, y1), (x2, y2)
            # 在图像上绘制矩形边界框
            cv2.rectangle(image, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
            # 计算字体厚度
            tf = max(tl - 1, 1)  # 字体厚度
            # 获取类别名称文本的尺寸
            t_size = cv2.getTextSize(cls_name, 0, fontScale=tl / 3, thickness=tf)[0]
            # 计算文本框的右下角坐标
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            # 在边界框上方绘制填充矩形以显示文本背景
            cv2.rectangle(image, c1, c2, color, -1, cv2.LINE_AA)  # 填充矩形
            # 在边界框上方显示类别名称和置信度
            cv2.putText(image, '{} {:.2f}'.format(cls_name, conf), (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255],
                        thickness=tf, lineType=cv2.LINE_AA)

        # 返回绘制了边界框的图像
        return image

    def detect(self, img):
        # 图像预处理，输出处理后的图像
        img0, img = self.preprocess(img)
        # 使用模型进行预测
        pred = self.m(img, augment=True)[0]
        # 将预测结果转换为float类型
        pred = pred.float()
        # 对预测结果应用非极大值抑制(NMS)，在NMS中使用设定的阈值
        pred = non_max_suppression(pred, self.threshold, 0.3)
        # 初始化用来存储检测到的框的列表
        pred_boxes = []
        # 用于存储图像信息的字典
        image_info = {}
        # 初始化检测到的物体计数
        count = 0
        # 初始化数字列表和计数器列表
        digital_list, meter_list = [], []

        # 遍历预测后的结果
        for det in pred:
            if det is not None and len(det):
                # 坐标缩放到原始图像尺寸
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
                # 对每一个检测到的物体框进行遍历
                for *x, conf, cls_id in det:
                    # 获取类别标签
                    lbl = self.names[int(cls_id)]
                    # 获取边框坐标，并转换为整型
                    x1, y1 = int(x[0]), int(x[1])
                    x2, y2 = int(x[2]), int(x[3])
                    # 根据边框坐标从原图中切割出相应的区域
                    region = img0[y1:y2, x1:x2]
                    # 根据类别标签分类计数器和数字区域
                    if lbl == "pointer":
                        meter_list.append(region)
                    else:
                        digital_list.append(region)
                    # 将检测到的边框添加到列表中
                    pred_boxes.append((x1, y1, x2, y2, lbl, conf))
                    # 更新计数
                    count += 1
                    # 为这个物体生成一个独特的键名
                    key = '{}-{:02}'.format(lbl, count)
                    # 将图像大小信息和置信度信息存储在字典中
                    image_info[key] = ['{}×{}'.format(x2 - x1, y2 - y1), np.round(float(conf), 3)]
        # 在图像上绘制所有检测到的边框
        im = self.plot_bboxes(img0, pred_boxes)
        # 返回绘制了边框的图像、图像信息、数字列表和计数器列表
        return im, image_info, digital_list, meter_list

# 测试代码
if __name__ == "__main__":
    detector = Detector()
    # 加载测试图像
    img_path = "output_images/新建文件夹/20240907_142306.jpg"
    img = cv2.imread(img_path)

    # 进行检测
    im, image_info, digital_list, meter_list = detector.detect(img)

    # 显示检测结果
    cv2.imshow("Detection", im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 打印检测列表和信息字典
    print("Meter list:", meter_list)
    print("Digital list:", digital_list)
    print("Image info:", image_info)

