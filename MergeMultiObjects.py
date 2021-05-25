# -*- coding: UTF-8 -*-

import os
import json
import random
from copy import deepcopy

import cv2
import numpy as np
from labelme import utils

import DataAugmentation # DataAugmentation.py


class MergeMultiObjects():
    """Merge [multiple objects] into [1 image].

    [Objects] has [.jpg/.png] + [.json] file    

    # Attributes:
    #     likes_spam: A boolean indicating if we like SPAM or not.
    #     eggs: An integer count of the eggs we have laid.
    """
    
    def __init__(self):      
        """ Parameters Settings """

        #INPUT/OUTPUT Data Directory    
        self.InputDIR = './input/single_object/'
        self.OutputDIR = './output/'

        #Merge Multiple Object Options
        self.merge_obj_num = 16            #[Number of objects] to merge in 1 image
        self.allow_same_obj_type = True    #Allow [same obj type] in 1 image
        
        #Load obj folder and obj filenames
        self.obj_type_list = []
        self.tot_obj_types = []
        self.obj_name_list_tot = []  #TODO:列出格式

        #Random selected img
        self.selected_img_list = []

        #Object mask, mask_color, json
        self.obj_mask_list = []
        self.obj_mask_color_list = []
        self.json_info_list = []

        #Final merged image
        self.img_merged = []

    def load_obj_filenames(self):
        """Load all object filenames"""

        self.obj_type_list = os.listdir(self.InputDIR)  
        self.tot_obj_types = len(self.obj_type_list)     
        
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
        
        print('================')
        print(' Objects info')
        print('================')
        print("Total object types: {}".format(len(self.obj_name_list_tot)))
        for iter_folder in range(len(self.obj_type_list)):
            obj_name = self.obj_type_list[iter_folder]
            tot_obj_images_num = len(self.obj_name_list_tot[iter_folder])
            print('Object type [{}]: {} images'.format(obj_name, tot_obj_images_num))  

    def select_obj_to_merge(self):
        """Randomly select [obj type] and [obj] to merge"""

        print('====================================')
        print(' Merge Multiple Objects Options')
        print('====================================')
        print('Total object types: {}'.format(self.tot_obj_types))
        print('Select ? objects to merge in 1 image: {}'.format(self.merge_obj_num))
        print('Allow [SAME obj type] in 1 image: {}'.format(self.allow_same_obj_type))

        #TODO:設定random seed
        # random.seed()
                
        #===========================#
        # Random select OBJECT TYPE  
        #===========================#
        if(self.allow_same_obj_type == True):             
            #obj可重複：一個obj_type可以選多個obj
            #TODO 檢查randint
            idx_obj_type_idx_list = [random.randint(0, self.tot_obj_types - 1) for _ in range(self.merge_obj_num)]

        else:             
            #obj不可重複：一個obj_type只選一個obj
            if(self.merge_obj_num <= self.tot_obj_types):                
                idx_obj_type_idx_list = random.sample(range(0, self.tot_obj_types), self.merge_obj_num)               
            else:
                print('ERROR! self.merge_obj_num should be SMALLER than self.tot_obj_types')
        
        idx_obj_type_idx_list.sort()

        #=======================#
        # Random select OBJECT  
        #=======================#
        for cnt in range(self.merge_obj_num):
            #random pick an object filename
            obj_type_idx = idx_obj_type_idx_list[cnt]            
            num_files_in_folder = len(self.obj_name_list_tot[obj_type_idx])
            file_idx = random.randint(0, num_files_in_folder - 1)            #TODO 檢查randint
            filename = self.obj_name_list_tot[obj_type_idx][file_idx]  
           
            #append filenames
            path = self.InputDIR + self.obj_type_list[obj_type_idx] + '/' + filename            
            self.selected_img_list.append(path)
            print(path)        

    def extract_obj(self):    
        """Extract obj mask from image and json file"""

        obj_mask_list = []
        obj_mask_color_list = []
        json_info_list = []

        for n in range(len(self.selected_img_list)):
            selected_img_path = self.selected_img_list[n] 
            print("Extract object from {}".format(selected_img_path))

            #============#
            # Read image
            #============#
            img_cv = cv2.imread(selected_img_path + '.png')   
            
            #======================#
            # Read and decode json
            #======================#
            toolhelper = DataAugmentation.ToolHelper()  
            json_info = toolhelper.parse_json(selected_img_path + '.json')
            json_info_list.append(json_info)  
            
            img = utils.img_b64_to_arr(json_info['imageData'])
            
            #建立每個物件對應的label值,不同類別不同像素值
            label_name_to_value = {'_background_': 0} #dictionary格式      
            
            for shape in json_info['shapes']:
                label_name = shape['label']
                if label_name in label_name_to_value: 
                    label_value = label_name_to_value[label_name]
                else:
                    label_value = len(label_name_to_value)
                    label_name_to_value[label_name] = label_value                    
            
            lbl, lbl_names = utils.shapes_to_label(img.shape, json_info['shapes'], label_name_to_value)

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
                all_mask_g = cv2.cvtColor(im_at_fixed, cv2.COLOR_GRAY2RGB) #im_at_fixed 單通道遮罩; all_mask_g 三通道遮罩
                imgObject = cv2.bitwise_and(all_mask_g, img_cv) #两个图片一个是彩色，另一个是彩色，不能做位与运算，需要将第一个图灰度话后再操作                
                
                obj_mask_list.append(all_mask_g)
                obj_mask_color_list.append(imgObject) 

        self.obj_mask_list = obj_mask_list
        self.obj_mask_color_list = obj_mask_color_list
        self.json_info_list = json_info_list

        return obj_mask_list, obj_mask_color_list, json_info_list
            
    def generate_single_channel_mask(self, obj_mask_color_list):
        obj_mask =[] 
        for size in range(len(obj_mask_color_list)):            
            retval, im_at_fixed = cv2.threshold(obj_mask_color_list[size][:,:,0], 0, 255, cv2.THRESH_BINARY)        
            all_mask_g = cv2.cvtColor(im_at_fixed, cv2.COLOR_GRAY2RGB)     
            obj_mask.append(all_mask_g)
        return obj_mask

    def merge_obj(self, obj_mask_list, obj_mask_color_list):     
        """Merge multiple object into 1 image"""

        height, width, _ = np.shape(obj_mask_color_list[0])
        img_merged = np.ones((height, width,3), np.uint8)
        for k in range(len(obj_mask_color_list)-1):
            if k == 0:
                mask_no_overlap = cv2.bitwise_xor(obj_mask_list[k], obj_mask_list[k+1])      #[未重疊]部份的遮罩
                obj1_no_overlap = cv2.bitwise_and(mask_no_overlap, obj_mask_color_list[k])   #第一物件[未重疊]部份
                obj2_no_overlap = cv2.bitwise_and(mask_no_overlap, obj_mask_color_list[k+1]) #第二物件[未重疊]部份

                mask_overlap = cv2.bitwise_and(obj_mask_list[k], obj_mask_list[k+1])         #[重疊]部份遮罩
                obj2_overlap = cv2.bitwise_and(mask_overlap, obj_mask_color_list[k+1])       #第二物件[重疊]部份
                
                img_merged = cv2.bitwise_or(obj1_no_overlap, obj2_no_overlap)                #第一物件[未重疊]+第二物件[未重疊]+第二物件[重疊]
                img_merged = cv2.bitwise_or(img_merged, obj2_overlap)

            else:
                mask_no_overlap = cv2.bitwise_xor(img_merged, obj_mask_list[k+1])            #[未重疊]部份的遮罩
                obj1_no_overlap = cv2.bitwise_and(mask_no_overlap, img_merged)               #新圖[未重疊]部份
                obj2_no_overlap = cv2.bitwise_and(mask_no_overlap, obj_mask_color_list[k+1]) #第k+1物件[未重疊]部份

                mask_overlap = cv2.bitwise_and(img_merged, obj_mask_list[k+1])               #[重疊]部份遮罩
                obj2_overlap = cv2.bitwise_and(mask_overlap, obj_mask_color_list[k+1])       #第k+1物件[重疊]部份
                
                img_merged = cv2.bitwise_or(obj1_no_overlap, obj2_no_overlap)                #新圖[未重疊]+第k+1物件[未重疊]+第k+1物件[重疊]
                img_merged = cv2.bitwise_or(img_merged, obj2_overlap)
        
        self.img_merged = img_merged
        print("Merge multiple objects COMPLETE!")
        return img_merged

if __name__ == '__main__':
    #=====================#
    # Parameters Settings    
    #=====================#
    input_folder_object = './input/single_object/'
    input_folder_background = './input/background/'
    output_folder = './output/'

    #===================#
    # MergeMultiObjects
    #===================#  
    mergeObj = MergeMultiObjects()
    mergeObj.InputDIR = input_folder_object
    mergeObj.OutputDIR = input_folder_background
    mergeObj.load_obj_filenames()
    mergeObj.select_obj_to_merge()
    
    #[擴增前]的單通道遮罩、三通道遮罩、json資料
    obj_mask_ori = []
    obj_mask_color_ori = []
    json_info_list_ori = []
    obj_mask_ori, obj_mask_color_ori, json_info_list_ori = mergeObj.extract_obj()
    
    #===================#
    # Data Augmentation
    #===================#
    #[擴增後]的單通道遮罩、三通道遮罩、json資料
    obj_mask_dataAug = []
    obj_mask_color_dataAug = []
    json_info_list_dataAug = []

    dataAug = DataAugmentation.DataAugmentForObjectDetection()
    for cnt in range(len(obj_mask_ori)):
        img_ori = obj_mask_color_ori[cnt]
        json_info_ori = json_info_list_ori[cnt]
        
        #Apply data augmentation to object
        img_dataAug, json_info_dataAug = dataAug.dataAugment(deepcopy(img_ori), deepcopy(json_info_ori))

        #Append image and json after data augmentation
        obj_mask_color_dataAug.append(img_dataAug)
        json_info_list_dataAug.append(json_info_dataAug)        

    obj_mask_dataAug = mergeObj.generate_single_channel_mask(obj_mask_color_dataAug)

    #===================#
    # MergeMultiObjects
    #===================#
    #Merge multi. obj. BEFORE data aug.
    merge_img = mergeObj.merge_obj(obj_mask_ori, obj_mask_color_ori)
    
    cv2.imwrite('merge_BEFORE_dataAug.jpg', merge_img)
    cv2.imshow('Merge multi. obj. BEFORE data aug.', merge_img)
    cv2.waitKey(0)
    
    #Merge multi. obj. AFTER data aug.
    merge_img_dataAug = mergeObj.merge_obj(obj_mask_dataAug, obj_mask_color_dataAug)
    
    cv2.imwrite('merge_AFTER_dataAug.jpg', merge_img_dataAug)
    cv2.imshow('Merge multi. obj. AFTER data aug.', merge_img_dataAug)
    cv2.waitKey(0)
    

    # #==========================#
    # # Save Image and JSON file
    # #==========================#
    # toolhelper = DataAugmentation.ToolHelper()

    # # save image
    # _file_prefix = 'test1'
    # cnt = 1
    # _file_suffix = '.jpg'
    # save_img_json_path = '../Data_Augmentation/'
    # img_name = '{}_{}{}'.format(_file_prefix, cnt + 1, _file_suffix)  
    # img_save_path = os.path.join(save_img_json_path, img_name)
    # toolhelper.save_img(img_save_path, merge_img_dataAug)  
    
    # # save json
    # ENCODING = 'utf-8'
    # raw_data = {}
    # raw_data["version"] = "4.5.6"
    # merge_shapes = toolhelper.concat_shapes(json_info_list_dataAug)  
    # raw_data["shapes"] = merge_shapes#data['shapes']
    # raw_data["imagePath"] = img_name
    # height, width, _ = np.shape(merge_img_dataAug)
    # raw_data["imageHeight"] = height
    # raw_data["imageWidth"] = width
    # base64_data = toolhelper.img2str(img_save_path)
    # raw_data["imageData"] = base64_data    

    # toolhelper.save_json('{}_{}.json'.format(_file_prefix, cnt + 1), save_img_json_path, raw_data)

