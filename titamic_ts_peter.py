# Import Libaries 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# load the data
train_data = pd.read_csv('train.csv')
pd_data = pd.read_csv('test.csv')

#print(train_data.head(10))
#print(train_data.describe())
#print(train_data.isna().sum())
# Drop the columns

print(pd_data)
# 填補 Fare 欄位的缺失值為其平均值
pd_data['Fare'] = pd_data['Fare'].fillna(pd_data['Fare'].mean())

# 填補 Age 欄位的缺失值為其平均值
pd_data['Age'] = pd_data['Age'].fillna(pd_data['Age'].mean())

print(pd_data)
PassengerId = pd_data['PassengerId']
train_data = train_data.drop(['Cabin','Name','PassengerId'], axis=1)
pd_data = pd_data.drop(['Cabin','Name','PassengerId'], axis=1)

print(pd_data)
train_data = train_data.drop(['Ticket'] , axis =1 )
pd_data = pd_data.drop(['Ticket'] , axis =1 )

# Remove the rows with missing values
train_data = train_data.dropna(subset =  ['Embarked','Age'])



#print(train_data.shape)
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()


# Encode the sex column
train_data.iloc[:, 2] = labelencoder.fit_transform(train_data.iloc[:, 2].values)
pd_data.iloc[:, 1] = labelencoder.fit_transform(pd_data.iloc[:, 1].values)



# Encode the embarked column
train_data.iloc[:, 7] = labelencoder.fit_transform(train_data.iloc[:, 7].values)
pd_data.iloc[:, 6] = labelencoder.fit_transform(pd_data.iloc[:, 6].values)

# Split the data into independent 'X' and dependent 'y' variables
X = train_data.iloc[:, 1:8].values
y = train_data.iloc[:, 0].values
# Split the dataset into 80% training and 20% testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
#Scale the data

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_trian = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


sc2 = StandardScaler()
pd_data = sc2.fit_transform(pd_data)

print(pd_data)

def models(X_train, y_train):
    # Use Dicision Tree
    from sklearn.tree import DecisionTreeClassifier
    tree = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
    tree.fit(X_train, y_train)
    print('[5]Decision Tree Training Accuracy:', tree.score(X_train, y_train))
    # Use the RandomForestClassifier
    #from sklearn.ensemble import RandomForestClassifier
    #forest = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
    #forest.fit(X_train,y_train)
    #return tree,forest
    return tree

model = models(X_train, y_train)


pred = model.predict(pd_data)
submission = pd.DataFrame({
    "PassengerId": PassengerId,
    "Survived": pred
})
submission.to_csv('submission.csv', index=False)

# Show the confusion matrix and accuracy for all of the models of the test data
"""from sklearn.metrics import confusion_matrix


for i in range( len(model) ):
    cm = confusion_matrix(y_test, model[i].predict(X_test))

    # Extract TN, FP, FN, TP
    TN, FP, FN, TP = confusion_matrix(y_test, model[i].predict(X_test)).ravel()
    
    test_score = (TP + TN) / (TN + TP + FP + FN)

    print(cm)
    print('Model[{}] Testing Accuracy = "{}"' .format(i, test_score))
    print()"""
