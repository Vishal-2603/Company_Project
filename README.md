README — YOLOv11 Custom Object Detection Project
Introduction

This project is part of my assessment task where I had to train an object detection model on a custom dataset. I chose YOLOv11, the latest version from Ultralytics, because it is fast, lightweight, and easy to fine-tune.

The main goal of the project was to build a model that can detect objects inside ID card images.
I used a dataset hosted on Roboflow, trained the model in Google Colab, and finally tested the trained weights on different images to check the performance.

What This Project Does

Downloads a custom dataset from Roboflow

Trains a YOLOv11 model using that dataset

Validates the model to check accuracy

Runs inference on images to detect objects

Shows outputs with bounding boxes and labels

The whole workflow is written in a single Python script so it’s easy to follow.

Why I Built It This Way

I wanted to create a clean and simple pipeline that shows:

I understand how dataset loading works

I can fine-tune YOLO models

I know how to perform validation

I can run inference and visualize results

I can work with different libraries like Roboflow, Ultralytics, Supervision, OpenCV, and PIL

This is the same process used in real-world computer vision tasks.

Tools & Libraries Used

Python

YOLOv11 (Ultralytics)

Roboflow API for dataset downloading

Supervision for drawing bounding boxes

OpenCV & PIL for image handling

Google Colab GPU for training

How to Run the Project

Install the dependencies:

pip install ultralytics supervision roboflow


Add your Roboflow API key in the Colab notebook (under Secrets).

Run the script to:

download the dataset

train the YOLO model

save the best weights

After training, the model weights are stored in:

runs/detect/train/weights/best.pt


Use these weights to run inference on any image.

Training Output

YOLO automatically generates:

Learning curves

Confusion matrix

Validation sample predictions

Best and last trained weights

These files show whether the model is improving and how well it performs.

Project Structure
92_yolo11_object_detection_on_custom_dataset.py   # main script
runs/detect/train/                                # training results
datasets/                                          # downloaded dataset

Conclusion

This project helped me understand the complete workflow of building a custom object detection model using YOLOv11.
The script includes training, validation, and prediction steps so it can be reused for any similar task.
