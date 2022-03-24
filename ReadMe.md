# Deep Learning - Mini Project 1
This is a submission for the Graduate Deep Learning Course at NYU Tandon. 


## Overview
The task is to use a resnet model with less than 5M parameters to maximize test accuracy on the CIFAR10 dataset. 


**Developers: Abhishek Rathod, Jake Gus, Utkarsh Shekhar**    
**Course: ECE-GY 7123 Spring 2022**

## Model Architechture
<img src="/figures/resnet18.png" width="600" height="300"/>

Given the starting template our architechture has the following parameters:

| Name | Value | Description |
| :---:        |     :---:      |         :---: |
|  N  |   4   |  Residual Layers |
| B    | [5, 3, 2, 1] | Residual blocks |
| C| 50| Channels in Residual Layer 1 |
| F| 3| Conv kernel size in residual layer |
| K| 1| Skip Connection kernel size  |
| P| 4| Average pool kernel size  |

This results in a total trainable parameters: **4.9M**

## Training

### Transforms
The following transforms are applied to the training set:

| Type | Arguments | Description |
| :---:        |     :---:      |         :---: |
|  Random Perspective  |   distortion = 0.3, p = 0.5   |  Performs a random perspective transformation of the given image with a given probability |
|  Random Crop |   size = 32, padding = 4  | Crop the given image at a random location|
|  Random Perspective  |   distortion = 0.3, p = 0.5   |  Performs a random perspective transformation of the given image with a given probability |
|  Random Horizontal Flip  |   p = 0.5   |  Horizontally flip the given image randomly with a given probability |
|  Normalize  |   (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  |  Normalize Images given mean and std |

The test set is only normalized using the same values. 

### Example of Transformed Images
<img src="/figures/transformed_images_example.png" width="1000" height="300"/>

The table below summarizes training paramters

| Type | Value | Arguments |
| :---:        |     :---:      |         :---: |
|  Train Batch Size  |   100  |  NA |
|  Test Batch Size  |   150  |  NA |
|  Max Epochs |   50  |  NA |
|  Optimizer  |   Adam  |  lr = 0.001, everything else default|
|  Scheduler  |   CosineAnnealingLR |  Tamx = max epochs, everything else default|


During training early stopping is implemented such that if the test loss begins to stagnate, training is stopped. 

## Results
<img src="/figures/accuracy_with_dropout_adam_4_9M.png" width="600" height="400"/>
<img src="/figures/loss_with_dropout_adam_4_9M.png" width="600" height="400"/>

### Accuracy Results By Class
| Class | Accuracy |
| :---:        |        :---: |
|plane | 92.4 |
|car   | 96.5 |
|bird  | 89.3 |
|cat   | 86.1 |
|deer  | 91.0 |
|dog   | 86.0 |
|frog  | 92.1 |
|horse | 92.7 |
|ship  | 94.3 |
|truck | 94.4 |


## Running the saved model
in main.py:
- lines 181, 183: Change 'load_model' to 'True' and 'epochs_to_run' to '1'

- Changes to the data can be made on lines 73-80

