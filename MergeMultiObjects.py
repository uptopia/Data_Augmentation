import os
import json
import random
import numpy as np
from copy import deepcopy

import cv2

from labelme import utils

import DataAugmentation # DataAugmentation.py

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
            toolhelper = DataAugmentation.ToolHelper()  
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

if __name__ == '__main__':
    data = MergeMultiObjects()
    data.select_obj_image()
    obj_mask, obj_mask_color, json_info_list = data.extract_obj()
    
    merge_img = data.merge_obj(obj_mask, obj_mask_color)
    cv2.imshow('before data_aug', merge_img)
    cv2.waitKey(0)
    cv2.imwrite('before data_aug.jpg', merge_img)

    dataAug = DataAugmentation.DataAugmentForObjectDetection()
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
    cv2.imwrite('after data_aug.jpg', merge_img_data_aug)

    toolhelper = DataAugmentation.ToolHelper()
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

