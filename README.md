
## T-shirt Object Detection

## Description

This project is an implementation of object detection to detect and segment a t-shirt in an image and measure the average color, height , width of the detected t-shirt.

## Requirements

    Python=3.8.10

All necessary packages are contained within the requirements.txt file. File path = /Neural Foundry/yolov5/requirements.txt

```sh
pip install -r requirements.txt
```

## Approach followed

Yolov5 object detection

## Dataset

1. Taken T-shirt dataset from roboflow. Here is the link https://universe.roboflow.com/yolo5tshirt/tshirt_detection . Dataset contains train, valid, test folders.

2. YOLOv5 format dataset folder structure.



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

4. All test, train, valid sample image in .jpg format of size 400x400, class- T-shirt.


## Model Training and Usage

```sh
python3 train.py --img 400 --batch 16 --epochs 30 --data dataset.yaml --weights yolov5s.pt --cache
```


The pre-trained weights yolov5s.pt were used to train the model on the dataset, with a batch size of 16 and 30 epochs. At the end of the training, two files, last.pt and best.pt, should be saved in the yolov5/runs/train/exp2/weights. The best.pt is used.

The results were saved to runs/train/exp2

## Model Testing

```sh
python3 detect.py --source runs/train/exp2/test3/ --weights best.pt --save-conf --save-txt
```
The detect.py script is being implemented with the best.pt weights. The results will be saved to runs/detect/exp9 and the labels will be saved to runs/detect/exp9/labels



## Model Evaluation

Used AP (Average Precision) - is an evaluation metric for object detection models. It is the average of the precision at different recall values, and is used to measure the overall accuracy of the model.

AP (Average Precision) score got 0.918 is generally considered good, as it indicates that the model is able to make accurate predictions with high precision.

Other metrics also used to evaluate model 

1. IOU
2. precision-recall curve


Color Detection - The opencv library was utilized to detect the average color of t-shirts, and the results were displayed in BGR format. Other methods that can be used to detect color include:

1. Color moments: Calculate the color moments (such as mean, standard deviation, skewness, and kurtosis) of the t-shirt region to determine the average color
2. Color quantization: Can use color quantization techniques such as the Median Cut or Octree methods to reduce the number of colors in the image and identify the dominant color
3. YOLOv5 image segmentation



## Conclusion

YOLOv5 is a state-of-the-art model for object detection on T-shirt dataset, known for its high accuracy and fast inference speed. However, the model can be further improved by using fine-tuning and data augmentation techniques to increase the diversity of the training set.

There are other approaches for object detection such as Faster R-CNN, RetinaNet etc. These approaches have their own strengths and weaknesses, and the choice of which one to use will depend on factors such as the specific application, the available computational resources, and the desired trade-off between speed and accuracy. In general, YOLOv5 is known for its real-time performance and high accuracy, making it a popular choice for object detection tasks.

Note: Given task at hand was completed based on the limited time and resources available. However, with more time and resources, the results could be further improved and optimized.


## References

1. https://doi.org/10.48550/arXiv.1506.02640
2. https://github.com/ultralytics/yolov5
3. https://universe.roboflow.com/yolo5tshirt/tshirt_detection
4. https://en.wikipedia.org/wiki/Object_detection
