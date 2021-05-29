# Data_Augmentation

##物件資料擴增＋合併多個物件＋更換背景：MergeMultiObjects.py
input/single_object
input/background
output/merge_objects
output/merge_dataAug_objects

===================================================
##其他使用方法
###更換背景：單獨使用ChangeDataBackground.py
input_folder
output_folder


###資料擴增：單獨使用DataAugmentation.py
input_folder
output_folder


```
Data_Augmentation
    ├── background
    │   ├── 1.jpg
    │   ├── ...
    │   └── 5.jpg
    ├── Extend_data
    │   ├── obj_data_9_8.jpg
    │   ├── obj_data_9_8.json
    │   ├── ...
    │   ├── obj_data_9_9.jpg
    │   └── obj_data_9_9.json
    ├── original_data
    │   ├── coffee1.json
    │   ├── coffee1.png
    │   ├── ...
    │   ├── hamburger1.json
    │   ├── hamburger1.png
    │   ├── ...    
    │   ├── sandwich1.jpg
    │   └── sandwich1.json
    ├── output_data
    │   ├── obj_data_8.jpg
    │   ├── obj_data_8.json
    │   ├── ...
    │   ├── obj_data_9.jpg
    │   └── obj_data_9.json
    └── png
        ├── coffee1.png
        ├── ...
        ├── hamburger1.png
        ├── ...    
        ├── lunchbox1.png
        ├── ...
        └── sandwich1.json
```

```
$cd ~
$git clone https://github.com/uptopia/Data_Augmentation.git
```

1. mkdir file (/background, /Extend_data, /original_data, /output_data, /png)

2. put your origin data to the /original_data folder. (include .jpg and .json)

3. put the background image which you want to change in the /background folder.

4. run ChangeDataBackground.py

5. run DataAugmentation.py    you will get the Augmentation data.

6. enjoy. 
