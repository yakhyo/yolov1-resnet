## YOLO: Real-Time Object Detection


<div align='center'>
  <img src='assets/person.jpg' height="250px">
  <img src='assets/person.jpg' height="250px">
</div>

**Train on voc2012+2007**

| Model                | Backbone | mAP@voc2007test  | FPS  |
| -------------------- | -------------- | ---------- | -------   |
| ResNet YOLOv1  |   ResNet50        | 0.65  |  __   |
| YOLOv1  |   darknet19        | 0.634      |  45   |


**Dataset:**
1. Download `voc2012train` [dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar)
2. Download `voc2007train` [dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar)
3. Download `voc2007test` [dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar)
4. Put all images in `JPEGImages` folder in `voc2012train` and `voc2007train` to `Images` folder as following:
```
├── Dataset 
    ├── Images
        ├── 1111.jpg
        ├── 2222.jpg
    ├── Labels
        ├── 1111.txt
        ├── 2222.txt
    ├── train.txt
    ├── test.txt
```

Each label consists of class and bounding box information. e.g `1xxx.txt` : 
```
1 255 247 425 468
0 470 105 680 468
1 152 356 658 754
```
**How to convert `.xml` files to `.txt` format?**
* Download [this repo](https://github.com/yakhyo/YOLO2VOC) and modify `config.py` to convert `VOC` format to `YOLO` format labels


**Train:**
- `python main.py`

**Evaluation:**
- `python eval.py`
- In `evaluation.py`, `im_show=False` change to `True` to see the results.

**Detection:**
 - To show the result image - `python detect.py --image assets/person.jpg`
 - Save result image - `python detect.py --image assets/person.jpg --save_img` 

**Weights:**
- Run the `download.sh` file in `weights` folder or download from this [link](https://www.dropbox.com/sh/nde76eig64rm02p/AADCumUHtwJgzyQeN2VvzBTxa?dl=0)

Result:
```text
CLASS                     AP
aeroplane                 0.71
bicycle                   0.72
bird                      0.71
boat                      0.59
bottle                    0.39
bus                       0.70
car                       0.78
cat                       0.83
chair                     0.43
cow                       0.69
diningtable               0.51
dog                       0.77
horse                     0.77
motorbike                 0.65
person                    0.74
pottedplant               0.37
sheep                     0.65
sofa                      0.57
train                     0.83
tvmonitor                 0.58
mAP: 0.65
```

**Reference:**
1. https://github.com/abeardear/pytorch-YOLO-v1
