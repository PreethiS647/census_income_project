

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

```
cat_cols = ['sex', 'education', 'marital-status', 'workclass', 'occupation']
cont_cols = ['age', 'hours-per-week']
y_col = ['label']

# RUN THIS CODE TO COMPARE RESULTS:
print(f'cat_cols  has {len(cat_cols)} columns')
print(f'cont_cols has {len(cont_cols)} columns')
print(f'y_col     has {len(y_col)} column')
```

<img width="357" height="87" alt="image" src="https://github.com/user-attachments/assets/a3f7e713-374a-412b-ad8b-c28a604f0c4a" />

```
# CODE HERE

for col in cat_cols:
    df[col] = df[col].astype('category')
# THIS CELL IS OPTIONAL
df = shuffle(df, random_state=101)
df.reset_index(drop=True, inplace=True)
df.head()
```

<img width="842" height="356" alt="image" src="https://github.com/user-attachments/assets/2e2173f9-a4a1-4ee9-ba47-769048064322" />

```
# CODE HERE
cat_szs = [len(df[col].cat.categories) for col in cat_cols]
emb_szs = [(size, min(50, (size+1)//2)) for size in cat_szs]
print(emb_szs)
```

<img width="557" height="81" alt="image" src="https://github.com/user-attachments/assets/3dcf1db5-532f-4b7d-a726-1cbe90932fbc" />

```
# CODE HERE

cats = np.stack([df[col].cat.codes.values for col in cat_cols], axis=1)

# RUN THIS CODE TO COMPARE RESULTS
cats[:5]
```

<img width="491" height="151" alt="image" src="https://github.com/user-attachments/assets/d755a1c6-8698-4b73-af9b-440a0ba9db37" />

```
# CODE HERE
cats = torch.tensor(cats, dtype=torch.int64)
print(cats[:5])
```

<img width="355" height="146" alt="image" src="https://github.com/user-attachments/assets/32d1ff33-ce31-4e0f-9db5-ac0f2b067907" />

```
# CODE HERE

conts = np.stack([df[col].values for col in cont_cols], axis=1)
conts[:5]

```

<img width="358" height="150" alt="image" src="https://github.com/user-attachments/assets/884e14df-9190-4ea6-8754-7a36a1d2ed2b" />

```
# CODE HERE
# RUN THIS CODE TO COMPARE RESULTS
conts = torch.tensor(conts, dtype=torch.float32)
print(conts.dtype)
```

<img width="208" height="70" alt="image" src="https://github.com/user-attachments/assets/7c9da578-6bfd-4d89-b0c0-56e0c609818c" />

```
# CODE HERE
y = torch.tensor(df[y_col].values, dtype=torch.int64).flatten()
# CODE HERE
b = 30000 # suggested batch size
t = 5000  # suggested test size

cat_train = cats[:b-t]
cat_test = cats[b-t:]
con_train = conts[:b-t]
con_test = conts[b-t:]
y_train = y[:b-t]
y_test = y[b-t:]

class TabularModel(nn.Module):

    def __init__(self, emb_szs, n_cont, out_sz, layers, p=0.5):
        # Call the parent __init__
        super().__init__()
        
        # Set up the embedding, dropout, and batch normalization layer attributes
        self.embeds = nn.ModuleList([nn.Embedding(ni, nf) for ni,nf in emb_szs])
        self.emb_drop = nn.Dropout(p)
        self.bn_cont = nn.BatchNorm1d(n_cont)
        
        # Assign a variable to hold a list of layers
        layerlist = []
        
        # Assign a variable to store the number of embedding and continuous layers
        n_emb = sum((nf for ni,nf in emb_szs))
        n_in = n_emb + n_cont
        
        # Iterate through the passed-in "layers" parameter (ie, [200,100]) to build a list of layers
        for i in layers:
            layerlist.append(nn.Linear(n_in,i)) 
            layerlist.append(nn.ReLU(inplace=True))
            layerlist.append(nn.BatchNorm1d(i))
            layerlist.append(nn.Dropout(p))
            n_in = i
        layerlist.append(nn.Linear(layers[-1],out_sz))
        
        # Convert the list of layers into an attribute
        self.layers = nn.Sequential(*layerlist)
    
    def forward(self, x_cat, x_cont):
        # Extract embedding values from the incoming categorical data
        embeddings = []
        for i,e in enumerate(self.embeds):
            embeddings.append(e(x_cat[:,i]))
        x = torch.cat(embeddings, 1)
        # Perform an initial dropout on the embeddings
        x = self.emb_drop(x)
        
        # Normalize the incoming continuous data
        x_cont = self.bn_cont(x_cont)
        x = torch.cat([x, x_cont], 1)
        
        # Set up model layers
        x = self.layers(x)
        return x
# CODE HERE
torch.manual_seed(33)
```
<img width="408" height="49" alt="image" src="https://github.com/user-attachments/assets/91ed182a-ba67-4a5e-bd2d-3777a6d6b8a6" />

```
# CODE HERE
model = TabularModel(emb_szs=emb_szs, n_cont=len(cont_cols), out_sz=2, layers=[50], p=0.4)
# RUN THIS CODE TO COMPARE RESULTS
model
```

<img width="910" height="453" alt="image" src="https://github.com/user-attachments/assets/db4b8459-2ec0-46cb-97a3-b3149863947f" />

```
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
import time
start_time = time.time()

epochs = 300
losses = []

for i in range(epochs):
    i+=1
    y_pred = model(cat_train, con_train)
    loss = criterion(y_pred, y_train)
    losses.append(loss)
    
    if i%25 == 1:
        print(f'epoch: {i:3}  loss: {loss.item():10.8f}')

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(f'epoch: {i:3}  loss: {loss.item():10.8f}') # print the last line
print(f'\nDuration: {time.time() - start_time:.0f} seconds') # print the time elapsed
```

<img width="370" height="393" alt="image" src="https://github.com/user-attachments/assets/d8befaee-2921-4b16-898b-97fd76694e1d" />


```
losses = [l.item() if torch.is_tensor(l) else l for l in losses]
plt.plot(np.array(losses, dtype=float))
plt.ylabel("Cross Entropy Loss")
plt.xlabel("Epoch")
plt.title("Model Training Loss")
plt.show()
```

<img width="727" height="572" alt="image" src="https://github.com/user-attachments/assets/0b14d731-dee5-4536-8764-272877480b7f" />

```
with torch.no_grad():
    y_val = model(cat_test, con_test)
    loss = criterion(y_val, y_test)

# RUN THIS CODE TO COMPARE RESULTS
print(f'CE Loss: {loss:.8f}')
```

<img width="305" height="60" alt="image" src="https://github.com/user-attachments/assets/0524ddf0-0681-47bb-b734-5490301a6c36" />

```
correct = 0
for i in range(len(y_test)):
    if torch.argmax(y_val[i]).item() == y_test[i].item():
        correct += 1
print(f"{correct} out of {len(y_test)} = {100 * correct / len(y_test):.2f}% correct")
```

<img width="393" height="65" alt="image" src="https://github.com/user-attachments/assets/cefe667f-c823-4c49-bca4-33fe0ac949d6" />


```
# WRITE YOUR CODE HERE:
def predict_income(model, encoders, cont_inputs, cat_inputs):
    model.eval()
    cat_tensor = torch.tensor([cat_inputs], dtype=torch.int64)
    cont_tensor = torch.tensor([cont_inputs], dtype=torch.float32)
    with torch.no_grad():
        output = model(cat_tensor, cont_tensor)
        pred = torch.argmax(output, dim=1).item()
    return pred
# Example mappings 
sex_d = {'Female': 0, 'Male': 1}
education_d = {
    '3': 0, '4': 1, '5': 2, '6': 3, '6': 4, '8': 5, '12': 6,
    'HS-grad': 7, 'Some-college': 8, 'Assoc-voc': 9, 'Assoc-acdm': 10, 'Bachelors': 11,
    'Masters': 12, 'Prof-school': 13, 'Doctorate': 14
}
marital_d = {'Divorced': 0, 'Married': 1, 'Married-spouse-absent': 2, 'Never-married': 3, 'Separated': 4, 'Widowed': 5}
workclass_d = {'Federal-gov': 0, 'Local-gov': 1, 'Private': 2, 'Self-emp': 3, 'State-gov': 4}
occupation_d = {'Adm-clerical': 0, 'Craft-repair': 1, 'Farming-fishing': 2, 'Handlers-cleaners': 3,
                'Machine-op-inspct': 4, 'Other-service': 5, 'Prof-specialty': 6, 'Protective-serv': 7,
                'Sales': 8, 'Tech-support': 9, 'Transport-moving': 10}

# Get inputs from user, convert and encode
age = int(input("What is the person's age? (18-90) "))
sex = sex_d[input("What is the person's sex? (Male/Female) ").capitalize()]
education = education_d[input("What is the person's education level? ").strip()]
marital = marital_d[input("What is the person's marital status? ").strip()]
workclass = workclass_d[input("What is the person's workclass? ").strip()]
occupation = occupation_d[input("What is the person's occupation? ").strip()]
hours_per_week = int(input("How many hours/week are worked? (20-90) "))

cat_inputs = [sex, education, marital, workclass, occupation]
cont_inputs = [age, hours_per_week]

predicted_label = predict_income(model, None, cont_inputs, cat_inputs)

print(f"\nThe predicted label is {predicted_label}")

```
<img width="488" height="254" alt="image" src="https://github.com/user-attachments/assets/08778885-3939-4f73-929b-89309ad22423" />














