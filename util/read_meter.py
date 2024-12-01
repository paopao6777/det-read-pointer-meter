import math
import os
import cv2
import numpy
import numpy as np
import torch
from skimage import morphology

class MeterReader(object):

    def __init__(self):
        pass

    def __call__(self, image,point_mask,dail_mask,word_mask,number,std_point):

        img_result = image.copy()
        value=self.find_lines(img_result,point_mask,dail_mask,number,std_point)
        print("value",value)

        return value



    def find_lines(self,ori_img,pointer_mask,dail_mask,number,std_point):

        # 实施骨架算法
        pointer_skeleton = morphology.skeletonize(pointer_mask)
        pointer_edges = pointer_skeleton * 255
        pointer_edges = pointer_edges.astype(np.uint8)
        # cv2.imshow("pointer_edges", pointer_edges)
        # cv2.waitKey(0)

        dail_mask = np.clip(dail_mask, 0, 1)
        dail_edges = dail_mask * 255
        dail_edges = dail_edges.astype(np.uint8)
        # cv2.imshow("dail_edges", dail_edges)
        # cv2.waitKey(0)

        pointer_lines = cv2.HoughLinesP(pointer_edges, 1, np.pi / 180, 10, np.array([]), minLineLength=10,
                                        maxLineGap=400)
        coin1, coin2 = None, None

        try:
            for x1, y1, x2, y2 in pointer_lines[0]:
                coin1 = (x1, y1)
                coin2 = (x2, y2)
                cv2.line(ori_img, (x1, y1), (x2, y2), (118, 198, 165), 2)
        except TypeError:
            return "can not detect pointer"


        h, w, _ = ori_img.shape
        center = (0.5 * w, 0.5 * h)
        dis1 = (coin1[0] - center[0]) ** 2 + (coin1[1] - center[1]) ** 2
        dis2 = (coin2[0] - center[1]) ** 2 + (coin2[1] - center[1]) ** 2
        if dis1 <= dis2:
            pointer_line = (coin1, coin2)
        else:
            pointer_line = (coin2, coin1)

        # print("pointer_line", pointer_line)

        if std_point==None:
            return "can not detect dail"

        # calculate angle
        a1 = std_point[0]
        a2 = std_point[1]
        cv2.circle(ori_img, a1, 5, (0, 115, 238), -1)
        cv2.circle(ori_img, a2, 5, (0, 115, 238), -1)
        one = [[pointer_line[0][0], pointer_line[0][1]], [a1[0], a1[1]]]
        two = [[pointer_line[0][0], pointer_line[0][1]], [a2[0], a2[1]]]
        three = [[pointer_line[0][0], pointer_line[0][1]], [pointer_line[1][0], pointer_line[1][1]]]
        # print("one", one)
        # print("two", two)
        # print("three",three)

        one=np.array(one)
        two=np.array(two)
        three = np.array(three)

        v1=one[1]-one[0]
        v2=two[1]-two[0]
        v3 = three[1] - three[0]

        distance=self.get_distance_point2line([a1[0], a1[1]],[pointer_line[0][0], pointer_line[0][1], pointer_line[1][0], pointer_line[1][1]])
        # print("dis",distance)

        flag=self.judge(pointer_line[0],std_point[0],pointer_line[1])
        # print("flag",flag)

        std_ang = self.angle(v1, v2)
        # print("std_result", std_ang)
        now_ang = self.angle(v1, v3)
        if flag >0:
            now_ang=360-now_ang
        # print("now_result", now_ang)


        # calculate value
        two_value = 40.0  # 将其固定为你想要的值，比如100

        # if number!=None and number[0]!="":
        #     two_value = float(number[0])
        # else:
        #     return "can not recognize number"
        if std_ang * now_ang !=0:
            value = (two_value / std_ang)
            value=value*now_ang
        else:
            return "angle detect error"

        if flag>0 and distance<40:
            value=0.00
        else:               
            value=round(value,3)

        font = cv2.FONT_HERSHEY_SIMPLEX

        # 获取图像的高度和宽度
        height, width = ori_img.shape[:2]

        # 设置文本的位置
        position = (10, height - 10)

        # 设置白色背景的矩形位置
        # 计算文本宽度和高度，给矩形留一些边距
        (text_width, text_height), baseline = cv2.getTextSize(str(value), font, 1.2, 2)
        bg_top_left = (position[0] - 5, position[1] - text_height - 5)  # 背景左上角坐标
        bg_bottom_right = (position[0] + text_width + 5, position[1] + baseline + 5)  # 背景右下角坐标

        # 画白色背景矩形
        cv2.rectangle(ori_img, bg_top_left, bg_bottom_right, (255, 255, 255), -1)

        # 在白色背景上绘制文本
        ori_img = cv2.putText(ori_img, str(value), position, font, 1.2, (0, 18, 25), 2)

        # cv2.imshow("result",ori_img)
        # cv2.waitKey(0)

        return value

    def get_distance_point2line(self, point, line):
        """
        Args:
            point: [x0, y0]
            line: [x1, y1, x2, y2]
        """
        line_point1, line_point2 = np.array(line[0:2]), np.array(line[2:])
        vec1 = line_point1 - point
        vec2 = line_point2 - point
        distance = np.abs(np.cross(vec1, vec2)) / np.linalg.norm(line_point1 - line_point2)
        return distance


    def judge(self,p1,p2,p3):
        A=p2[1]-p1[1]
        B=p1[0]-p2[0]
        C=p2[0]*p1[1] - p1[0]*p2[1]

        value=A*p3[0] + B*p3[1] +C

        return value


    def angle(self,v1, v2):
        lx=np.sqrt(v1.dot(v1))
        ly=np.sqrt(v2.dot(v2))
        cos_angle=v1.dot(v2) / (lx * ly)

        angle=np.arccos(cos_angle)
        angle2=angle*360 / 2 / np.pi

        return angle2



if __name__ == '__main__':
    tester = MeterReader()
    root = 'demo'
    for image_name in os.listdir(root):
        print(image_name)
        path = f'{root}/{image_name}'
        image = cv2.imread(path)
        result = tester(image)
        print(result)
        # cv2.imshow('a', image)
        # cv2.waitKey()
