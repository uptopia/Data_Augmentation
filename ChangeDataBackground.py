import json
import matplotlib.pyplot as plt
import cv2
import os
from labelme import utils
import numpy as np
from base64 import b64encode
from json import dumps

import time
import random
from skimage.util import random_noise
import base64
import json
import re
from copy import deepcopy
import argparse
import math


class DataAugmentation():
    def __init__(self):
        #=====================#
        # Parameters Settings    
        #=====================#
        self.rename = 'obj_data'
        self.obj_number = 50
        self.cnt = 0
        
        # INPUT Data Directory
        self.InputDIR = './original_data/' #'./Extend_data/'
        self.BackgroundDIR = './background/'
        
        # OUTPUT Data Directory
        self.PNGDIR = './png/'
        self.OutputDIR = './output_data/'

        self.imgWidth = 640
        self.imgHeight = 480

        #================#
        # Read Filenames    
        #================#
        self.datanames1 = os.listdir(self.InputDIR)
        self.datanames2 = os.listdir(self.BackgroundDIR)

        self.json_data = np.zeros([self.obj_number,1]).astype(np.str)
        self.json_name=np.zeros([self.obj_number,1]).astype(np.str)
        self.oldbackground_data = np.zeros([self.obj_number,1]).astype(np.str)        
        self.newbackground_data = np.zeros([self.obj_number,1]).astype(np.str)        
        self.output_data = np.zeros([self.obj_number,1]).astype(np.str)        
        
        i = 0
        j = 0
        m = 0

        for dataname in self.datanames1:
            if os.path.splitext(dataname)[1] == '.json': #original_data目录下包含.json的文件
                self.json_data[i] = dataname
                print("aaa,".join(self.json_data[i]))
                self.json_name[i], suffix = os.path.splitext(",".join(self.json_data[i]))
                i+=1
              
        for dataname in self.datanames1:
            if os.path.splitext(dataname)[1] == '.png': #original_data目录下包含.png/.jpg的文件
                self.oldbackground_data[m] = dataname
                m+=1

        for dataname in self.datanames2:
            if os.path.splitext(dataname)[1] == '.jpg': #background目录下包含.jpg的文件
                self.newbackground_data[j] = dataname
                j+=1
               
    def _changeLight(self, img):
        alpha = random.uniform(0.35, 1)
        blank = np.zeros(img.shape, img.dtype)
        return cv2.addWeighted(img, alpha, blank, 1 - alpha, 0)

    def generate_data(self):
        for k in range(len([name for name in os.listdir(self.BackgroundDIR) if os.path.isfile(os.path.join(self.BackgroundDIR, name))])):

            newbackground = cv2.imread(self.BackgroundDIR + '/{}'.format(",".join(self.newbackground_data[k])))  # 读取图片img2
            newbackground = cv2.resize(newbackground, (self.imgWidth, self.imgHeight))  # 为图片重新指定尺寸
            newbackground = cv2.cvtColor(newbackground,cv2.COLOR_BGR2RGB)

            for j in range(int(len(([name for name in os.listdir(self.InputDIR) if os.path.isfile(os.path.join(self.InputDIR, name))]))/2)): 
                #===================#
                # Decode .json file
                #===================#
                json_file = self.InputDIR +'/{}.json'.format(",".join(self.json_name[j]))
                data = json.load(open(json_file))
                img = utils.img_b64_to_arr(data['imageData'])
                label_name_to_value = {'_background_': 0}
                for shape in data['shapes']:
                    label_name = shape['label']                    
                    if label_name in label_name_to_value:
                        label_value = label_name_to_value[label_name]                        
                    else:
                        label_value = len(label_name_to_value)
                        label_name_to_value[label_name] = label_value                        
                        
                lbl, lbl_names = utils.shapes_to_label(img.shape, data['shapes'], label_name_to_value)

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
                
                cv2.imwrite(self.PNGDIR + "/{}.png".format(",".join(self.json_name[j])), all_mask) 

                #=====================#
                # Label, InverseLabel    
                #=====================#
                label = cv2.imread( self.PNGDIR + '/{}.png'.format(",".join(self.json_name[j])))
                label = cv2.resize(label, (self.imgWidth, self.imgHeight))

                inv_label = 255 - label

                #================#
                # Extract Object    
                #================#
                oldbackground = cv2.imread(self.InputDIR +'/{}.png'.format(",".join(self.json_name[j]))) #oldbackground_data-->json_name
                oldbackground = cv2.resize(oldbackground, (self.imgWidth, self.imgHeight))
                oldbackground = cv2.cvtColor(oldbackground, cv2.COLOR_BGR2RGB)        

                imgObject = cv2.bitwise_and(label, oldbackground) #两个图片一个是彩色，另一个是彩色，不能做位与运算，需要将第一个图灰度话后再操作
        
                #=======================#
                # Extract NewBackground    
                #=======================#
                newbackground = self._changeLight(newbackground)

                imgBackground = cv2.bitwise_and(inv_label, newbackground)
        
                #===================================#
                # NewImage = imgObject + NewBackground
                #===================================#
                imgTogether = imgObject + imgBackground
                imgTogether = cv2.cvtColor(imgTogether, cv2.COLOR_BGR2RGB)

                #=======================#
                # Save .JPG .JSON files
                #=======================#
                # save image                                
                cv2.imwrite(self.OutputDIR + "/{}_{}.jpg".format(self.rename, self.cnt + 1),imgTogether) #cv2.imwrite(self.OutputDIR + "/{}.jpg".format(",".join(self.json_name[j])),imgTogether)

                # save json
                ENCODING = 'utf-8'
                IMAGE_NAME = '{}_{}.jpg'.format(self.rename, self.cnt + 1) #IMAGE_NAME = '{}.jpg'.format(",".join(self.json_name[j])) 
                JSON_NAME = '{}_{}.json'.format(self.rename, self.cnt + 1) #JSON_NAME = '{}.json'.format(",".join(self.json_name[j]))
                
                with open(self.OutputDIR + IMAGE_NAME, 'rb') as jpg_file:
                    byte_content = jpg_file.read()
                base64_bytes = b64encode(byte_content)
                base64_string = base64_bytes.decode(ENCODING)

                raw_data = {}
                raw_data["version"] = "4.5.6"
                raw_data["shapes"] = data['shapes']
                raw_data["imagePath"] = IMAGE_NAME
                raw_data["imageHeight"] = self.imgHeight
                raw_data["imageWidth"] = self.imgWidth
                raw_data["imageData"] = base64_string
                
                jsondata = dumps(raw_data, indent = 2)
                
                with open(self.OutputDIR + JSON_NAME, 'w') as json_file:
                    json_file.write(jsondata)

                print('Generating dataset : {}_{}.jpg'.format(self.rename, self.cnt + 1))
                print('Generating dataset : {}_{}.json'.format(self.rename, self.cnt + 1) )

                self.cnt+=1

if __name__ == '__main__':
    data = DataAugmentation()
    data.generate_data()
    
