import pandas as pd

x = pd.read_csv('x.txt')
y = pd.read_csv('y.txt')

# Print head and row, column dimensions
print(x.shape)
print(x.head())

print(y.shape)
print(y.head())
