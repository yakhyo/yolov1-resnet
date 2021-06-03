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



**Reference:**
1. 

Our map in voc2007 test set is 0.665~ some result are below, you can see more in testimg folder.

