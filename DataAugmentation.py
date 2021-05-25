# encoding='UTF-8'

import time
import random
import cv2
import os
import numpy as np
from skimage.util import random_noise
import base64
import json
import re
from copy import deepcopy
import argparse
import math


class DataAugmentForObjectDetection():
    def __init__(self, change_light_rate=0.5,
                 add_rotate=0.5, add_noise_rate=0.5, random_point=0.5, flip_rate=0.5, shift_rate=0.5, rand_point_percent=0.03,
                 is_addrotate=True, is_addNoise=True, is_changeLight=True, is_random_point=True, is_shift_pic_bboxes=True,
                 is_filp_pic_bboxes=True):
        
        self.rotate_angle = 10

        self.add_rotate = add_rotate
        self.change_light_rate = change_light_rate
        self.add_noise_rate = add_noise_rate
        self.random_point = random_point
        self.flip_rate = flip_rate
        self.shift_rate = shift_rate
        self.rand_point_percent = rand_point_percent

        self.is_addrotate = is_addrotate
        self.is_addNoise = is_addNoise
        self.is_changeLight = is_changeLight
        self.is_random_point = is_random_point
        self.is_filp_pic_bboxes = is_filp_pic_bboxes
        self.is_shift_pic_bboxes = is_shift_pic_bboxes

    # 加噪聲
    def _addNoise(self, img):
        return random_noise(img, seed=int(time.time())) * 255

    # 調整亮度
    def _changeLight(self, img):
        alpha = random.uniform(0.35, 1)
        blank = np.zeros(img.shape, img.dtype)
        return cv2.addWeighted(img, alpha, blank, 1 - alpha, 0)

    # 隨機的改變點的值
    def _addRandPoint(self, img):
        percent = self.rand_point_percent
        num = int(percent * img.shape[0] * img.shape[1])
        for i in range(num):
            rand_x = random.randint(0, img.shape[0] - 1)
            rand_y = random.randint(0, img.shape[1] - 1)
            if random.randint(0, 1) == 0:
                img[rand_x, rand_y] = 0
            else:
                img[rand_x, rand_y] = 255
        return img

    # 旋轉
    def _addrotate(self, img, angle, json_info):
        #image
        h, w, _ = img.shape
        center = (w/2, h/2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotate_img = cv2.warpAffine(img, M, (w, h))

        #json
        shapes = json_info['shapes']
        for shape in shapes:
            for p in shape['points']:
                p[0] = center[0] + (math.cos(angle*math.pi/180)*(p[0] - center[0]) - math.sin(angle*math.pi/180)*(center[1] - p[1]))
                p[1] = center[1] - (math.sin(angle*math.pi/180)*(p[0] - center[0]) + math.cos(angle*math.pi/180)*(center[1] - p[1]))
        return rotate_img, json_info


    # 平移
    def _shift_pic_bboxes(self, img, json_info):

        # ---------------------- 平移图像 ----------------------
        h, w, _ = img.shape
        x_min = w
        x_max = 0
        y_min = h
        y_max = 0

        shapes = json_info['shapes']
        for shape in shapes:
            points = np.array(shape['points'])
            x_min = min(x_min, points[:, 0].min())
            y_min = min(y_min, points[:, 1].min())
            x_max = max(x_max, points[:, 0].max())
            #y_max = max(y_max, points[:, 0].max())
            y_max = max(y_max, points[:, 1].max())

        d_to_left = x_min  # 包含所有目标框的最大左移动距离
        d_to_right = w - x_max  # 包含所有目标框的最大右移动距离
        d_to_top = y_min  # 包含所有目标框的最大上移动距离
        d_to_bottom = h - y_max  # 包含所有目标框的最大下移动距离

        x = random.uniform(-(d_to_left - 1) / 3, (d_to_right - 1) / 3)
        y = random.uniform(-(d_to_top - 1) / 3, (d_to_bottom - 1) / 3)

        M = np.float32([[1, 0, x], [0, 1, y]])  # x为向左或右移动的像素值,正为向右负为向左; y为向上或者向下移动的像素值,正为向下负为向上
        shift_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

        # ---------------------- 平移boundingbox ----------------------
        for shape in shapes:
            for p in shape['points']:
                p[0] += x
                p[1] += y
        return shift_img, json_info

    # 镜像
    def _filp_pic_bboxes(self, img, json_info):

        # ---------------------- 翻转图像 ----------------------
        h, w, _ = img.shape

        sed = random.random()

        if 0 < sed < 0.33:  
            flip_img = cv2.flip(img, 0)  # _flip_x
            inver = 0
        elif 0.33 < sed < 0.66:
            flip_img = cv2.flip(img, 1)  # _flip_y
            inver = 1
        else:
            flip_img = cv2.flip(img, -1)  # flip_x_y
            inver = -1

        # ---------------------- 调整boundingbox ----------------------
        shapes = json_info['shapes']
        for shape in shapes:
            for p in shape['points']:
                if inver == 0:
                    p[1] = h - p[1]
                elif inver == 1:
                    p[0] = w - p[0]
                elif inver == -1:
                    p[0] = w - p[0]
                    p[1] = h - p[1]

        return flip_img, json_info

    # 图像增强方法
    def dataAugment(self, img, dic_info):

        change_num = 0  # 改变的次数
        while change_num < 1:  # 至少有一種數據增强生效
            
            if self.is_changeLight:
                if random.random() > self.change_light_rate:  
                    change_num += 1
                    img = self._changeLight(img)
            '''
            if self.is_addNoise:
                if random.random() < self.add_noise_rate:  
                    change_num += 1
                    img = self._addNoise(img)
            '''
            if self.is_addrotate:
                if random.random() < self.add_rotate:
                    change_num += 1
                    angle = self.rotate_angle
                    img, dic_info = self._addrotate(img, angle, dic_info)
            
            '''        
            if self.is_random_point:
                if random.random() < self.random_point:  
                    change_num += 1
                    img = self._addRandPoint(img)
            '''
            
            if self.is_shift_pic_bboxes:
                if random.random() < self.shift_rate:  
                    change_num += 1
                    img, dic_info = self._shift_pic_bboxes(img, dic_info)
                    
            if self.is_filp_pic_bboxes or 1:
                if random.random() < self.flip_rate: 
                    change_num += 1
                    img, bboxes = self._filp_pic_bboxes(img, dic_info)

        return img, dic_info


# xml解析工具
class ToolHelper():

    def parse_json(self, path):
        with open(path)as f:
            json_data = json.load(f)
        return json_data

    # 對圖片進行編碼
    def img2str(self, img_name):
        with open(img_name, "rb")as f:
            base64_data = str(base64.b64encode(f.read()))
        match_pattern = re.compile(r'b\'(.*)\'')
        base64_data = match_pattern.match(base64_data).group(1)
        return base64_data

    def save_img(self, save_path, img):
        cv2.imwrite(save_path, img)


    def save_json(self, file_name, save_folder, dic_info):
        with open(os.path.join(save_folder, file_name), 'w') as f:
            json.dump(dic_info, f, indent=2)

    def concat_shapes(self, dic_info_list):
        total_shapes = []
        for n in range(len(dic_info_list)):
            json_info = dic_info_list[n]
            shapes = json_info['shapes']
            for shape in shapes:
                total_shapes.append(shape)
        return total_shapes


if __name__ == '__main__':

    need_aug_num = 10  #需要擴增的次數

    toolhelper = ToolHelper()  

    is_endwidth_dot = True  # 文件是否以.jpg或者png结尾

    dataAug = DataAugmentForObjectDetection()

    parser = argparse.ArgumentParser()
    parser.add_argument('--source_img_json_path', type = str, default = '../Data_Augmentation/output_data') #input
    parser.add_argument('--save_img_json_path', type = str, default = '../Data_Augmentation/Extend_data')   #output
    args = parser.parse_args()
    source_img_json_path = args.source_img_json_path
    save_img_json_path = args.save_img_json_path  

    # 如果保存文件夹不存在就创建
    if not os.path.exists(save_img_json_path):
        os.mkdir(save_img_json_path)

    for parent, _, files in os.walk(source_img_json_path):
        print('parent: {}, files: {}'.format(parent, files))
        files.sort()  # 排序一下
        print('files sorted', files)
        for file in files:
            # load image, json files
            if file.endswith('jpg') or file.endswith('png'):
                cnt = 0
                pic_path = os.path.join(parent, file)
                json_path = os.path.join(parent, file[:-4] + '.json')
                json_dic = toolhelper.parse_json(json_path)

                if is_endwidth_dot:
                    dot_index = file.rfind('.')
                    _file_prefix = file[:dot_index]  # 文件名的前缀
                    _file_suffix = file[dot_index:]  # 文件名的後缀
                img = cv2.imread(pic_path)

                while cnt < need_aug_num:
                    # data augmentation
                    auged_img, json_info = dataAug.dataAugment(deepcopy(img), deepcopy(json_dic))
                
                    # save image
                    img_name = '{}_{}{}'.format(_file_prefix, cnt + 1, _file_suffix)  
                    img_save_path = os.path.join(save_img_json_path, img_name)
                    toolhelper.save_img(img_save_path, auged_img)  
                    
                    # save json
                    json_info['imagePath'] = img_name
                    base64_data = toolhelper.img2str(img_save_path)
                    json_info['imageData'] = base64_data
                    toolhelper.save_json('{}_{}.json'.format(_file_prefix, cnt + 1), save_img_json_path, json_info)

                    #print(img_name)
                    cnt += 1