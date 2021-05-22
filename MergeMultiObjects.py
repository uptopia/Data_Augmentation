import os
# import re
# import argparse
# import math
# import time
import random

import cv2

import json
# from json import dumps
# import base64
# from base64 import b64encode

import numpy as np
from labelme import utils
# from copy import deepcopy
# import matplotlib.pyplot as plt
# from skimage.util import random_noise

class MergeMultiObjects():
    def __init__(self):
        #=====================#
        # Parameters Settings    
        #=====================#
        # INPUT/OUTPUT Data Directory    
        self.InputDIR = '/home/upup/Documents/objects_input/'           
        self.OutputDIR = '/home/upup/Documents/objects_output/'

        self.merge_type_num = 5

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
        # #可重複
        # rand_x = [random.randint(0, self.tot_obj_types - 1) for _ in range(self.merge_type_num)]
        
        self.selected_img = []
        #不可重複
        idx_type = random.sample(range(0, self.tot_obj_types - 1), self.merge_type_num)
        
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
        cnt=0    
        for n in range(len(self.selected_img)):
            path1 = self.selected_img[n] 
            print(path1)
            img = cv2.imread(path1 + '.png')
            
            #read json
            toolhelper = ToolHelper()  
            json_info = toolhelper.parse_json(path1 + '.json') #'/home/upup/Downloads/obj_data_1_1.json')#path1+'.json') #

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
                cnt=cnt+1
                cv2.imshow('im_at_fixed', im_at_fixed)
                cv2.waitKey(0)

                all_mask_g = cv2.cvtColor(im_at_fixed, cv2.COLOR_GRAY2RGB)
                imgObject = cv2.bitwise_and(all_mask_g, img) #两个图片一个是彩色，另一个是彩色，不能做位与运算，需要将第一个图灰度话后再操作
                obj_mask.append(imgObject) 

        height, width, _ = np.shape(imgObject)
        fin_img = np.ones((height, width,3), np.uint8)
        for k in range(len(obj_mask)-1):

            if k == 0:
                fin_img = cv2.bitwise_xor(obj_mask[k], obj_mask[k+1])
            else:
                fin_img = cv2.bitwise_xor(fin_img, obj_mask[k+1])
            
            cv2.imshow('111fin_img', fin_img)
            cv2.waitKey(0)
        cv2.imwrite('fin_img.png', fin_img)


# xml解析工具
class ToolHelper():

    def parse_json(self, path):
        with open(path)as f:
            json_data = json.load(f)
        return json_data

if __name__ == '__main__':
    data = MergeMultiObjects()
    data.select_obj_image()
    data.extract_obj()




