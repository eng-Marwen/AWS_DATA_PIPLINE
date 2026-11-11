import pandas as pd
df = pd.read_csv("data_set.csv")
print(df.shape)               # about (42372, 1037)
print(df.columns[-1])         # should be the label (FLAG)
print(df.iloc[:, -1].value_counts())   # about 38.7k zeros, 3.6k ones