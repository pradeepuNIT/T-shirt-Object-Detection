
## T-shirt Object Detection

## Description

This project is an implementation of object detection to detect and segment a t-shirt in an image and measure the average color, height , width of the detected t-shirt.

## Requirements

    Python
    OpenCV
    numpy
    torch
    YOLOv5 & others

requirements.txt contains all necessary packages. File path = /Neural Foundry/yolov5/requirements.txt

```sh
pip install -r requirements.txt
```

## Approach followed

Yolov5 object detection

## Dataset

1. I have taken dataset of t-shirt from roboflow. Here is the link https://universe.roboflow.com/yolo5tshirt/tshirt_detection . Dataset contains train, valid, test folders of t-shirt images.

2. Changed dataset folder structure in to yolov5 format as show below.

     dataset
        - images
          - train
            - img1.jpg
            - img2.jpg
            ...
            ...
            
          - validation
            - img1.jpg
            - img2.jpg
            ...
            ...

        - labels
          - train
            - img1_label.txt
            - img2_label.txt
            ...
            ...

          - validation
            - img1_label.txt
            - img2_label.txt

        - dataset.yaml
          # train and val data
          train: ../dataset/images/train/
          val: ../dataset/images/validation/

          # number of classes
          nc: 1

          # class names
          names: ['T-shirt']


3. Total train samples in dataset = 430
   Total test  samples in dataset = 61
   Total valid  samples in dataset= 123

4. All test, train, valid samples image are .jpg format of size 400x400 and using only one class i.e: 't-shirt'


## Model Training and Usage

```sh
python3 train.py --img 400 --batch 16 --epochs 30 --data dataset.yaml --weights yolov5s.pt --cache
```


yolov5s.pt pre-trained weights used to train model on dataset, batch size 16 with 30 epocs.At the end of the training, two files should be saved in yolov5/runs/train/exp2/weights: last.pt and best.pt. I used best.pt.

Results saved to runs/train/exp2

## Model Testing

```sh
python3 detect.py --source runs/train/exp2/test3/ --weights best.pt --save-conf --save-txt
```
Implementing detect.py script with best.pt weights. The results will be saved to runs/detect/exp9 and labels to runs/detect/exp9/labels



## Model Evaluation

Used AP (Average Precision) - is an evaluation metric for object detection models. It is the average of the precision at different recall values, and is used to measure the overall accuracy of the model.

AP (Average Precision) score got 0.918 is generally considered good, as it indicates that the model is able to make accurate predictions with high precision.

Other metrics also used to evaluate model 

1. IOU
2. precision-recall curve


Color Detection - I have used opencv to detect avg color of t-shirt and results are showed in BGR format.There are other methods which can be performed to detec color. those are: 

1. Color moments: You can calculate the color moments (such as mean, standard deviation, skewness, and kurtosis) of the t-shirt region to determine the average color.
2. Color quantization: You can use color quantization techniques such as the Median Cut or Octree methods to reduce the number of colors in the image and identify the dominant color.
3. YOLOv5 image segmentation



## Conclusion

YOLOv5 for object detection on a T-shirt dataset is a state-of-the-art model that is known for its high accuracy and fast inference speed. The model can be improved by fine tuning and data augmentation techniques to increase the diversity of the training set.

There are other approaches for object detection like Faster R-CNN, RetinaNet etc. All of these approaches have their own strengths and weaknesses, and the choice of which one to use will depend on factors such as the specific application, the available computational resources, and the desired trade-off between speed and accuracy.In general YOLOv5 is known for its real-time performance and high accuracy so it can be used more in object detections problems.

Note: Given task is completed based on limited available time and resources. It can be improved even much better without earlier limitations.


## References

1. https://doi.org/10.48550/arXiv.1506.02640 YOLO Algorithm 
2. https://github.com/ultralytics/yolov5
3. https://universe.roboflow.com/yolo5tshirt/tshirt_detection
4. https://en.wikipedia.org/wiki/Object_detection
