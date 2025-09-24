import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def makeLinearDS():
    x = np.array([ 
        [2, 3],
        [1, 5],
        [2, 6],
        [6, 2],
        [7, 3],
        [8, 2],
        [7, 6],
        [6, 7],
        [5, 8],
        [8, 7]])
    y = np.array([-1, -1, -1, 1, 1, 1, 1, 1, -1, -1])
    return x,y

def makeNonLinearDS():
    x = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
        [0.5, 0.5],
        [1, 0.5],
        [0.5, 1],
        [0.25, 0.75],
        [0.75, 0.25],
        [0.6, 0.6]
    ])
    y = np.array([-1, 1, 1, -1, 1, -1, -1, 1, 1, -1])
    return x, y

#titanic set
def loadTitanicDS(path = r'COMP379HW2\DATA\titanicTrain.csv', test_size=0.3, random_state=4):
    df = pd.read_csv(path)
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    df = df[features + ['Survived']]
    
    #handles missing values
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

    df['Sex'] = df['Sex'].map({'male':0, 'female':1})
    df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)


    x = df.drop('Survived', axis=1).values
    y = df['Survived'].values

    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
    return x_train, x_test, y_train, y_test