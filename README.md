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

The full implementation is in `notebooks/census_income_workshop.ipynb`.

---

## DATASET INFORMATION
- Dataset: `income.csv` (30,000 entries)
- Columns include age, fnlwgt, education_num, capital_gain, capital_loss, hours_per_week, and income_level.
- The dataset must be placed in the project root folder or downloaded as instructed in the notebook.

---

## OUTPUT

- **Training Loss vs Epochs Plot**  
  _Shows how the loss decreases over 300 epochs._

- **Accuracy on Test Set**  
  _Displays the final classification accuracy._

- **Sample Prediction**  
  _Shows model output for a user-provided sample input._

---

## PROJECT STRUCTURE
income_project" 
