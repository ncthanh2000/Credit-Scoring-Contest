import pandas as pd
train = pd.read_excel(r'Final Data/Train - Std Data.xlsx')
test = pd.read_excel(r'Final Data/Test - Std Data.xlsx')

print(len(train.columns))
print(train.columns)
print("-"*30)
print(len(test.columns))
print(test.columns)

intersection = list(set(train.columns) & set(test.columns))
print("Intersect between train and test")
print(intersection)

train.drop(columns = intersection, inplace = True)
test.drop(columns = intersection, inplace = True)

print('-'*30)
print('Mixmatch Columns - Train')
print(len(train.columns))
print(train.columns)
print("-"*30)
print('Mixmatch Columns - Test')
print(len(test.columns))
print(test.columns)