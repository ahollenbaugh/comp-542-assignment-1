import pandas as pd
import numpy as np

# Read in the dataset:
dataset = pd.read_csv("C:\\Users\\aghol\\OneDrive\\Desktop\\COMP 542\\Assignment1\\drug200.csv", sep=',')

# Convert categorical data into numerical data:
from sklearn.preprocessing import LabelEncoder
# Identify which features contain categorical data:
categorical_cols = [col for col, dtype in dataset.dtypes.items() if dtype == 'object']
# print(categorical_cols)
# Assign an integer to each category:
le = LabelEncoder()
for col in categorical_cols:
    dataset[col] = le.fit_transform(dataset[col])
print(dataset)

 