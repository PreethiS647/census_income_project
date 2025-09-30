

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



