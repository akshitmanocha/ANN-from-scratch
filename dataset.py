import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

train_data = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
test_data = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
X_train = train_data.iloc[:, 1:] 
Y_train = train_data['label']
X_test = test_data.iloc[:, 1:]
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=42,shuffle=True)

if __name__ == "__main__":
    print("Train Size:", X_train.shape)
    print("Validation Size:", X_val.shape)