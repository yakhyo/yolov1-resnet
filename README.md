## YOLO: Real-Time Object Detection


<div align='center'>
  <img src='assets/person.jpg' height="250px">
  <img src='assets/person.jpg' height="250px">
</div>

**Train on voc2012+2007**

| Model                | Backbone | mAP@voc2007test  | FPS  |
| -------------------- | -------------- | ---------- | -------   |
| ResNet YOLOv1  |   ResNet50        | _training_   |  ___   |
| YOLOv1  |   darknet19        | 63.4%      |  45   |

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
- `python evaluation.py`
- In `evaluation.py`, `im_show=False` change to `True` to see the results.

Result:
```text
CLASS                     AP
aeroplane                 0.59
bicycle                   0.60
bird                      0.50
boat                      0.31
bottle                    0.19
bus                       0.66
car                       0.59
cat                       0.77
chair                     0.24
cow                       0.55
diningtable               0.42
dog                       0.70
horse                     0.66
motorbike                 0.60
person                    0.54
pottedplant               0.23
sheep                     0.48
sofa                      0.44
train                     0.71
tvmonitor                 0.54
mAP: 0.52
```

**Reference:**
1. https://github.com/abeardear/pytorch-YOLO-v1
