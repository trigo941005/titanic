# Import Libaries 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# load the data
train_data = pd.read_csv('train.csv')
#print(train_data.head(10))
#print(train_data.describe())
#print(train_data.isna().sum())
# Drop the columns
train_data = train_data.drop(['Cabin','Name','PassengerId'], axis=1)
train_data = train_data.drop(['Ticket'] , axis =1 )
# Remove the rows with missing values
train_data = train_data.dropna(subset =  ['Embarked','Age'])
#print(train_data.shape)
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()

# Encode the sex column
train_data.iloc[:, 2] = labelencoder.fit_transform(train_data.iloc[:, 2].values)

# Encode the embarked column
train_data.iloc[:, 7] = labelencoder.fit_transform(train_data.iloc[:, 7].values)
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
print(X_test)
X_test = sc.transform(X_test)
print(X_test)
# Use SVC (linear kernal)


def models(X_train, y_train):

    # Use Logistic Regression
    from sklearn.linear_model import LogisticRegression
    log = LogisticRegression(random_state = 0)
    log.fit(X_train, y_train)

    # Use KNeighbors 
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors= 5, metric = 'minkowski', p = 1)
    knn.fit(X_train, y_train)

    # Use SVC (linear kernal)
    from sklearn.svm import SVC
    svc_lin = SVC(kernel='linear', random_state = 0)
    svc_lin.fit(X_train,y_train)

    # Use SVC (RBF kernal)
    from sklearn.svm import SVC
    svc_rbf = SVC(kernel='rbf', random_state = 0)
    svc_rbf.fit(X_train,y_train)

    # Use GaussianNB 
    from sklearn.naive_bayes import GaussianNB
    gauss = GaussianNB()
    gauss.fit(X_train, y_train)

    # Use Dicision Tree
    from sklearn.tree import DecisionTreeClassifier
    tree = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
    tree.fit(X_train, y_train)

    # Use the RandomForestClassifier
    from sklearn.ensemble import RandomForestClassifier
    forest = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
    forest.fit(X_train,y_train)
    
    #Print the training accuracy for each model
    print('[0]Logistic Regression Training Accuracy:', log.score(X_train, y_train))
    print('[1]K Neighbors  Training Accuracy:', knn.score(X_train, y_train))
    print('[2]SVC Linear Training Accuracy:', svc_lin.score(X_train, y_train))
    print('[3]SVC RBF Training Accuracy:', svc_rbf.score(X_train, y_train))
    print('[4]Gaussian NB Training Accuracy:', gauss.score(X_train, y_train))
    print('[5]Decision Tree Training Accuracy:', tree.score(X_train, y_train))
    print('[6]Random Forest Training Accuracy:', forest.score(X_train, y_train))
    return log,knn,svc_lin,svc_rbf,gauss,tree,forest
# Get the train all of the models
model = models(X_train, y_train)

# Show the confusion matrix and accuracy for all of the models of the test data
from sklearn.metrics import confusion_matrix

for i in range( len(model) ):
    cm = confusion_matrix(y_test, model[i].predict(X_test))

    # Extract TN, FP, FN, TP
    TN, FP, FN, TP = confusion_matrix(y_test, model[i].predict(X_test)).ravel()
    
    test_score = (TP + TN) / (TN + TP + FP + FN)

    print(cm)
    print('Model[{}] Testing Accuracy = "{}"' .format(i, test_score))
    print()
