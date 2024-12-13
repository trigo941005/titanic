import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# load the data
test_data = pd.read_csv('test.csv')
test_data = test_data.drop(['Cabin','Name','PassengerId'], axis=1)
test_data = test_data.drop(['Ticket'] , axis =1 )
test_data = test_data.dropna(subset =  ['Fare','Age'])
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
test_data.iloc[:, 1] = labelencoder.fit_transform(test_data.iloc[:, 1].values)

# Encode the embarked column
test_data.iloc[:, 6] = labelencoder.fit_transform(test_data.iloc[:, 6].values)
X = test_data.iloc[:, 0:7].values
#print(X)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_test = sc.fit_transform(X)
#print(test_data)
print(X_test)