

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








