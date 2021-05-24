import os
import re
# import argparse
import math
import time
import random

import cv2

import json
# from json import dumps
import base64
from base64 import b64encode

import numpy as np
from labelme import utils
from copy import deepcopy
# import matplotlib.pyplot as plt
from skimage.util import random_noise

class MergeMultiObjects():
    def __init__(self):
        #=====================#
        # Parameters Settings    
        #=====================#
        # INPUT/OUTPUT Data Directory    
        self.InputDIR = '/home/upup/Documents/objects_input/'           
        self.OutputDIR = '/home/upup/Documents/objects_output/'

        self.merge_type_num = 10

        #=======================#
        # Read object filenames  
        #=======================#
        self.obj_type_list = os.listdir(self.InputDIR)  
        self.tot_obj_types = len(self.obj_type_list)     

        self.obj_name_list_tot = []
        for iter_folder in range(len(self.obj_type_list)):            
            obj_folder = self.InputDIR + self.obj_type_list[iter_folder] #current folder       
            obj_files_list = os.listdir(obj_folder)                      #files in current folder

            obj_name_list = []
            for iter_file in range(len(obj_files_list)):
                filename = obj_files_list[iter_file]

                name, suffix = os.path.splitext(filename)
                if os.path.splitext(filename)[1] == '.png':
                    obj_name_list.append(name)
            
            obj_name_list.sort()            
            self.obj_name_list_tot.append(obj_name_list)

        #Print objects info
        print('================')
        print('  Objects info')
        print('================')
        print("Total object types: {}".format(len(self.obj_name_list_tot)))
        for iter_folder in range(len(self.obj_type_list)):
            obj_name = self.obj_type_list[iter_folder]
            tot_obj_images_num = len(self.obj_name_list_tot[iter_folder])
            print('Object type [{}]: {} images'.format(obj_name, tot_obj_images_num))        

    def select_obj_image(self):
        print('select {} types of objects'.format(self.merge_type_num))
        # random.seed()
        #可重複
        idx_type = [random.randint(0, self.tot_obj_types - 1) for _ in range(self.merge_type_num)]
        
        self.selected_img = []
        # #不可重複
        # idx_type = random.sample(range(0, self.tot_obj_types - 1), self.merge_type_num)
        
        for cnt in range(self.merge_type_num):            
            folder = self.InputDIR + self.obj_type_list[idx_type[cnt]]            
            files_cnt = len(self.obj_name_list_tot[idx_type[cnt]])            
            idx_file = random.randint(0, files_cnt - 1)            
            filename = self.obj_name_list_tot[idx_type[cnt]][idx_file]            
            path = folder + '/' + filename# +'.png'
            # print(path)
            self.selected_img.append(path)
        print(self.selected_img)


    def extract_obj(self):
        # read image
        obj_mask = []
        obj_mask_color = []
        json_info_list = []

        cnt=0    
        for n in range(len(self.selected_img)):
            path1 = self.selected_img[n] 
            print(path1)
            img_cv = cv2.imread(path1 + '.png')
   
            
            #read json
            toolhelper = ToolHelper()  
            json_info = toolhelper.parse_json(path1 + '.json') #'/home/upup/Downloads/obj_data_1_1.json')#path1+'.json') #
            json_info_list.append(json_info)

            #===================#
            # Decode .json file
            #===================#        
            data = json_info
            img = utils.img_b64_to_arr(data['imageData'])
            
            #建立每個物件對應的label值,不同類別不同像素值
            label_name_to_value = {'_background_': 0} #dictionary格式      
            
            for shape in data['shapes']:
                label_name = shape['label']
                if label_name in label_name_to_value: 
                    label_value = label_name_to_value[label_name]
                else:
                    label_value = len(label_name_to_value)
                    label_name_to_value[label_name] = label_value                    
            
            lbl, lbl_names = utils.shapes_to_label(img.shape, data['shapes'], label_name_to_value)
            print(lbl.shape)
            print(lbl_names.shape)

            mask = []
            class_id = []
            for i in range(1, len(label_name_to_value)):
                mask.append((lbl==i).astype(np.uint8)) 
                class_id.append(i)
            mask = np.asarray(mask,np.uint8)
            mask = np.transpose(np.asarray(mask,np.uint8),[1,2,0])
    
            all_mask = 0
                        
            for i in range(0, len(class_id)):
                retval, im_at_fixed = cv2.threshold(mask[:,:,i], 0, 255, cv2.THRESH_BINARY) 
                all_mask = all_mask + im_at_fixed
                print('mask{}.png'.format(cnt))
                cv2.imwrite('mask{}.png'.format(cnt), im_at_fixed)
                cnt = cnt + 1

                all_mask_g = cv2.cvtColor(im_at_fixed, cv2.COLOR_GRAY2RGB) #im_at_fixed 單通道遮罩; all_mask_g 三通道遮罩
                print(im_at_fixed.shape)
                print(all_mask_g.shape)
                imgObject = cv2.bitwise_and(all_mask_g, img_cv) #两个图片一个是彩色，另一个是彩色，不能做位与运算，需要将第一个图灰度话后再操作                
                obj_mask.append(all_mask_g)
                obj_mask_color.append(imgObject) 

        self.obj_mask = obj_mask
        self.obj_mask_color = obj_mask_color
        self.json_info_list = json_info_list

        return obj_mask, obj_mask_color, json_info_list

    def merge_obj(self, obj_mask, obj_mask_color):
   
        obj_mask =[] 
        for size in range(len(obj_mask_color)):            
            retval, im_at_fixed = cv2.threshold(obj_mask_color[size][:,:,0], 0, 255, cv2.THRESH_BINARY)        
            all_mask_g = cv2.cvtColor(im_at_fixed, cv2.COLOR_GRAY2RGB)     
            obj_mask.append(all_mask_g)
            
        height, width, _ = np.shape(obj_mask_color[0])
        fin = np.ones((height, width,3), np.uint8)
        for k in range(len(obj_mask_color)-1):
            if k == 0:
                # fin_img_color = cv2.bitwise_xor(obj_mask_color[k], obj_mask_color[k+1])

                mask_no_overlap = cv2.bitwise_xor(obj_mask[k], obj_mask[k+1])           #[未重疊]部份的遮罩
                obj1_no_overlap = cv2.bitwise_and(mask_no_overlap, obj_mask_color[k])   #第一物件[未重疊]部份
                obj2_no_overlap = cv2.bitwise_and(mask_no_overlap, obj_mask_color[k+1]) #第二物件[未重疊]部份

                mask_overlap = cv2.bitwise_and(obj_mask[k], obj_mask[k+1])              #[重疊]部份遮罩
                obj2_overlap = cv2.bitwise_and(mask_overlap, obj_mask_color[k+1])       #第二物件[重疊]部份
                
                fin = cv2.bitwise_or(obj1_no_overlap, obj2_no_overlap)                  #第一物件[未重疊]+第二物件[未重疊]+第二物件[重疊]
                fin = cv2.bitwise_or(fin, obj2_overlap)
                # cv2.imshow('fin_tmp', fin)
                # cv2.waitKey(0)
            else:
                # fin_img_color = cv2.bitwise_xor(obj_mask_color[k], obj_mask_color[k+1])

                mask_no_overlap = cv2.bitwise_xor(fin, obj_mask[k+1])                   #[未重疊]部份的遮罩
                obj1_no_overlap = cv2.bitwise_and(mask_no_overlap, fin)                 #新圖[未重疊]部份
                obj2_no_overlap = cv2.bitwise_and(mask_no_overlap, obj_mask_color[k+1]) #第k+1物件[未重疊]部份

                mask_overlap = cv2.bitwise_and(fin, obj_mask[k+1])                      #[重疊]部份遮罩
                obj2_overlap = cv2.bitwise_and(mask_overlap, obj_mask_color[k+1])       #第k+1物件[重疊]部份
                
                fin = cv2.bitwise_or(obj1_no_overlap, obj2_no_overlap)                  #新圖[未重疊]+第k+1物件[未重疊]+第k+1物件[重疊]
                fin = cv2.bitwise_or(fin, obj2_overlap)
                # cv2.imshow('fin_tmp', fin)
                # cv2.waitKey(0)

        # cv2.imshow('fin', fin)
        # cv2.waitKey(0)
        # cv2.imwrite('fin.png', fin)

        return fin

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
        print('addNoise')
        return random_noise(img, seed=int(time.time())) * 255

    # 調整亮度
    def _changeLight(self, img):
        alpha = random.uniform(0.35, 1)
        blank = np.zeros(img.shape, img.dtype)
        print('change_light: alpha = {}'.format(alpha))
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
        print('addRandPoint')
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
        
        print('rotate: angle = {}'.format(angle))
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
        
        print('shift image: (delta_x, delta_y) = ({}, {})'.format(x,y))
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
        print('flip: inver = {}'.format(inver))
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
    data = MergeMultiObjects()
    data.select_obj_image()
    obj_mask, obj_mask_color, json_info_list = data.extract_obj()
    
    merge_img = data.merge_obj(obj_mask, obj_mask_color)
    cv2.imshow('before data_aug', merge_img)
    cv2.waitKey(0)

    dataAug = DataAugmentForObjectDetection()
    for cnt in range(len(obj_mask)):
        img_ori = obj_mask_color[cnt]
        json_info_ori = json_info_list[cnt]
        
        auged_img, json_info = dataAug.dataAugment(deepcopy(img_ori), deepcopy(json_info_ori))

        #replace original data
        obj_mask_color[cnt] = auged_img
        json_info_list[cnt] = json_info

    merge_img_data_aug = data.merge_obj(obj_mask, obj_mask_color)
    cv2.imshow('after data_aug', merge_img_data_aug)
    cv2.waitKey(0)

    toolhelper = ToolHelper()
    # save image
    _file_prefix = 'test1'
    cnt = 1
    _file_suffix = '.jpg'
    save_img_json_path = '../Data_Augmentation/'
    img_name = '{}_{}{}'.format(_file_prefix, cnt + 1, _file_suffix)  
    img_save_path = os.path.join(save_img_json_path, img_name)
    toolhelper.save_img(img_save_path, merge_img_data_aug)  
    
    # save json
    ENCODING = 'utf-8'
    raw_data = {}
    raw_data["version"] = "4.5.6"
    merge_shapes = toolhelper.concat_shapes(json_info_list)  
    raw_data["shapes"] = merge_shapes#data['shapes']
    raw_data["imagePath"] = img_name
    height, width, _ = np.shape(merge_img_data_aug)
    raw_data["imageHeight"] = height
    raw_data["imageWidth"] = width
    base64_data = toolhelper.img2str(img_save_path)
    raw_data["imageData"] = base64_data    

    toolhelper.save_json('{}_{}.json'.format(_file_prefix, cnt + 1), save_img_json_path, raw_data)

