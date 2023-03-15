# Create dataset for training

## **Objective**
- The popular method is to use a CNN model which succeeds in a similar problem. 
- ImageNet is the large image data set for image classification problems. 
- We use ResNet-50, starting with pretrained data for ImageNet and finetuning the model for fitting the problem here. 



## **Overview**

### ***Sequence***
- Input
  - train and test sets as a form of torch.utils.data.DataLoader
  - number of epochs - how many loops the model trains the training set
  - device name - specify the name of gpu
- Output
  - model after training



## **Details**

### **Libraries needed**

```python
import numpy as np
import os
from sklearn.model_selection import StratifiedShuffleSplit
import torchvision.transforms as transforms
import torch
import torchvision
```