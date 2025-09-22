# Census Income Project

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
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import random

# Set random seed
SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)

# Load dataset
df = pd.read_csv(r"C:\Users\admin\Downloads\income.csv.zip", compression='zip')

# Continuous columns
cont_columns = ['age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
# Label column
label_column = 'income_level'

# Convert continuous columns to tensors
cont_data = torch.tensor(df[cont_columns].values, dtype=torch.float32)

# Encode labels
labels = torch.tensor(LabelEncoder().fit_transform(df[label_column]), dtype=torch.long)

# Split into training and testing sets
cont_train, cont_test, y_train, y_test = train_test_split(
    cont_data, labels, train_size=25000, test_size=5000, random_state=SEED
)
```
## model design
```
import torch.nn as nn
import torch.nn.functional as F

class TabularModel(nn.Module):
    def __init__(self, n_cont):
        super().__init__()
        self.bn = nn.BatchNorm1d(n_cont)  # BatchNorm for continuous features
        self.fc1 = nn.Linear(n_cont, 50)  # Hidden layer with 50 neurons
        self.dropout = nn.Dropout(0.4)    # Dropout p=0.4
        self.fc2 = nn.Linear(50, 2)       # Output layer for binary classification

    def forward(self, x):
        x = self.bn(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

n_cont = cont_train.shape[1]
model = TabularModel(n_cont)
```
## training
import torch.optim as optim

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
```
epochs = 300
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(cont_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 50 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

```
  <img width="238" height="127" alt="image" src="https://github.com/user-attachments/assets/4d0206ad-0e5c-4f72-b417-ed23c3ea9f6f" />

## evaluation
```
model.eval()
with torch.no_grad():
    outputs = model(cont_test)
    preds = torch.argmax(outputs, dim=1)
    accuracy = (preds == y_test).sum().item() / len(y_test)
    test_loss = criterion(outputs, y_test).item()

print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy:.4f}")
```
<img width="365" height="44" alt="image" src="https://github.com/user-attachments/assets/2abdddef-aa2a-440b-9897-ba9e99ae22c3" />

## BONUS: Predict New Data
def predict_new(model, cont_values):
    model.eval()
    cont_tensor = torch.tensor([cont_values], dtype=torch.float32)
    with torch.no_grad():
        output = model(cont_tensor)
        pred = torch.argmax(output, dim=1).item()
    return ">50K" if pred == 1 else "<=50K"

# Example usage
new_data = [35, 200000, 13, 0, 0, 40]  # age, fnlwgt, education_num, capital_gain, capital_loss, hours_per_week
print(predict_new(model, new_data))
```
