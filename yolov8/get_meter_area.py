import torch
import numpy as np
from yolov8.ultralytics.nn.tasks import attempt_load_weights
from yolov8.ultralytics.utils.ops import non_max_suppression, scale_coords
from yolov8.ultralytics.data.augment import LetterBox
from yolov8.ultralytics.utils.torch_utils import select_device
import cv2
from random import randint
import os
import time


class Detector(object):

    def __init__(self):
        self.img_size = 640
        self.threshold = 0.6
        self.max_frame = 160
        self.init_model()

    def init_model(self):
        self.weights = 'yolov8/best.pt'
        self.device = '0' if torch.cuda.is_available() else 'cpu'
        self.device = select_device(self.device)
        model = attempt_load_weights(self.weights, device=self.device)  # 尝试获取元组的第一个元素
        model.to(self.device).eval()
        model.half()  # 只有当你确定你的模型支持FP16时才能调用
        self.m = model
        self.names = model.module.names if hasattr(model, 'module') else model.names
        # print(type(self.names), self.names)

        self.colors = [(randint(0, 255), randint(0, 255), randint(0, 255)) for _ in self.names]

    def preprocess(self, img):
        img0 = img.copy()  # 对原始图像进行复制，以便于后续处理不影响原图
        # 实例化一个LetterBox对象，用于对图像进行尺寸调整
        letterbox = LetterBox(new_shape=self.img_size)
        # print(letterbox)
        # 直接从调用letterbox()结果中获取img
        img = letterbox(image=img0)
        # 将图像从BGR转换到RGB格式，神经网络模型常用RGB格式
        img = img[:, :, ::-1].transpose(2, 0, 1)  # 这里的img是处理后的图像
        # 保证图像数据在内存中连续存储，为接下来转换为Tensor做准备
        img = np.ascontiguousarray(img)
        # 将图像数据从NumPy数组转换为PyTorch Tensor，并发送到指定的设备(CPU或GPU)
        img = torch.from_numpy(img).to(self.device)
        # 将图像数据类型转换为半精度浮点数，这样有助于加速计算，但要求硬件支持
        img = img.half()  # 半精度
        # 将图像像素值进行归一化处理，归一化在[0,1]范围内
        img /= 255.0
        # 如果图像是单通道，添加一个额外的批次维度，便于模型处理
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # 返回处理前的原图和处理后的图像张量
        return img0, img

    def plot_bboxes(self, image, bboxes, line_thickness=None):
        tl = line_thickness or round(
            0.002 * (image.shape[0] + image.shape[1]) / 2) + 1  # 线条/字体厚度
        for (x1, y1, x2, y2, cls_name, conf) in bboxes:
            # 我们得到的cls_name是一个字符串，是类的名字。
            # cls_id是字典self.names的键，而cls_name是对应的值。
            # 因此，我们需要找到cls_name对应的cls_id。
            cls_id = None
            for key, value in self.names.items():
                if value == cls_name:  # 比较找出cls_name对应的ID
                    cls_id = key
                    break

            if cls_id is None:  # 如果没有找到对应的类别ID，跳过这个bbox
                continue

            color = self.colors[cls_id]  # 使用找到的cls_id来获取颜色
            c1, c2 = (x1, y1), (x2, y2)
            cv2.rectangle(image, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
            tf = max(tl - 1, 1)  # 字体厚度
            t_size = cv2.getTextSize(cls_name, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(image, c1, c2, color, -1, cv2.LINE_AA)  # 填充背景
            cv2.putText(image, f'{cls_name} {conf:.2f}', (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255],
                        thickness=tf, lineType=cv2.LINE_AA)
        return image

    # def plot_bboxes(self, image, bboxes, line_thickness=None):
    #     # 如果没有指定线条粗细，则根据图像大小动态计算线条粗细
    #     tl = line_thickness or round(0.002 * (image.shape[0] + image.shape[1]) / 2) + 1
    #     # 遍历所有的边界框(bboxes)参数，每个bbox包含了起始点(x1, y1)、终点(x2, y2)、类别名(cls_name)和置信度(conf)
    #     for (x1, y1, x2, y2, cls_name, conf) in bboxes:
    #         # 从字典中获取类名对应的id，如果类名不存在，则返回-1
    #         cls_id = self.names.get(cls_name, -1)
    #         if cls_id == -1:  # 如果类名不存在字典中，则跳过该框的绘制
    #             continue
    #         # 获取该类别对应的颜色
    #         color = self.colors[cls_id]
    #         # 起始点和终点坐标，用于绘制矩形
    #         c1, c2 = (x1, y1), (x2, y2)
    #         # 在图像上绘制边界框
    #         cv2.rectangle(image, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    #         # 文字标签的字体大小（确保至少为1）
    #         tf = max(tl - 1, 1)
    #         # 获取类名文字的大小
    #         t_size = cv2.getTextSize(cls_name, 0, fontScale=tl / 3, thickness=tf)[0]
    #         # 计算文字背景的顶点坐标
    #         c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
    #         # 绘制文本背景的矩形
    #         cv2.rectangle(image, c1, c2, color, -1, cv2.LINE_AA)
    #         # 在图像上绘制类名和置信度
    #         cv2.putText(image, '{} {:.2f}'.format(cls_name, conf), (c1[0], c1[1] - 2), 0, tl / 3,
    #                     [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    #     # 返回绘制完成的图像
    #     return image

    def detect(self, im, i):
        # 图像预处理，输出处理后的图像
        im0, img = self.preprocess(im)
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
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                # 对每一个检测到的物体框进行遍历
                for *x, conf, cls_id in det:
                    # 获取类别标签
                    lbl = self.names[int(cls_id)]
                    # 获取边框坐标，并转换为整型
                    x1, y1 = int(x[0]), int(x[1])
                    x2, y2 = int(x[2]), int(x[3])
                    # 根据边框坐标从原图中切割出相应的区域
                    region = im0[y1:y2, x1:x2]
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
        im = self.plot_bboxes(im, pred_boxes)
        # 返回绘制了边框的图像、图像信息、数字列表和计数器列表
        return im, image_info, digital_list, meter_list


if __name__ == "__main__":
    det = Detector()
    path = 'demo/'
    img_list = os.listdir(path)
    total_time = 0
    num = 0

    for i in img_list:
        img = cv2.imread(path + i)
        cv2.imshow(f'read - {i}', img)
        cv2.waitKey(0)

        start_time = time.time()

        # 使用 detect 方法获取结果
        detected_image, image_info, digital_list, meter_list = det.detect(img, i)

        end_time = time.time()
        total_time += end_time - start_time
        fps = (num + 1) / total_time
        num += 1
        print("FPS:", fps)

        # 打印检测到的对象信息
        print("Image Information:", image_info)

        # 打印数字列表和计数器列表的长度，表示检测到了多少个对象
        print("Number of Digits Detected:", len(digital_list))
        print("Number of Meters Detected:", len(meter_list))

        # 显示绘制了检测框的图像
        cv2.imshow(f'Detected - {i}', detected_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()




