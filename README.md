<img width="197" height="50" alt="image" src="https://github.com/user-attachments/assets/107c34a5-09cc-4891-a0cf-6db32b8f537e" /># Census Income Project

## AIM
To develop a binary classification neural network model for predicting whether an individual earns more than $50,000 annually using the Census Income dataset.

## THEORY
Classification problems involve predicting discrete class labels based on input features. Traditional models may struggle with mixed types of data (categorical and continuous). Neural networks, particularly feedforward networks with embeddings for categorical variables and batch-normalized continuous inputs, can effectively model complex relationships in tabular data. In this project, a neural network model is trained using PyTorch for binary classification.

## Neural Network Model
The model uses:
- Embeddings for categorical features
- Batch-normalized continuous features
- One hidden layer with 50 neurons
- Dropout with p=0.4  
- Output layer with 2 neurons for binary classification  

---

## DESIGN STEPS

### STEP 1: Data Preparation
- Separate categorical, continuous, and label columns.
- Encode categorical features and labels using LabelEncoder.
- Convert features and labels into PyTorch tensors.
- Split the dataset: 25,000 training samples, 5,000 testing samples.

### STEP 2: Define the Neural Network Model
- Create a `TabularModel` class in PyTorch.
- Input: embeddings + batch-normalized continuous features
- Hidden Layer: 50 neurons, ReLU activation, dropout p=0.4
- Output Layer: 2 neurons (binary classification)

### STEP 3: Define Loss Function and Optimizer
- Loss Function: CrossEntropyLoss
- Optimizer: Adam, learning rate = 0.001

### STEP 4: Train the Model
- Set random seed for reproducibility.
- Train for 300 epochs.
- Track training loss at each epoch.

### STEP 5: Evaluate the Model
- Evaluate on the test set.
- Report test loss and accuracy.

### STEP 6 (BONUS): Predict New Data
- Create a function to input new data (e.g., hours-per-week, education, marital status) and output the model’s prediction.

---

## PROGRAM

### Name: PREETHI S  
### Register Number: 212223230157

## Program


## Data preparation
```
import torch
import torch.nn as nn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
%matplotlib inline

df = pd.read_csv('income.csv')
print(len(df))
df.head()

```
<img width="1150" height="248" alt="image" src="https://github.com/user-attachments/assets/5bad292e-28af-4abe-97b4-72a50131b635" />
```
df['label'].value_counts()
```

<img width="381" height="109" alt="image" src="https://github.com/user-attachments/assets/eda89e26-3a06-4b59-ad76-0b011e377bcf" />
```
df.columns
```
<img width="743" height="112" alt="image" src="https://github.com/user-attachments/assets/50999990-e4ab-4284-b381-207bef1830c7" />



## result:
The neural network model successfully predicts whether an individual earns more than $50,000 annually, achieving good accuracy on the test set. New inputs can be classified using the trained model.



