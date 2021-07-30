import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

class Multiple_Regression():
    def __init__(self,dataset,oneHotencoding):
        self.dataset = dataset
        self.oneHotencoding = oneHotencoding

    def import_Index(self):
        # Importing the dataset we are going to use
        dataset = pd.read_csv(self.dataset)
        x = dataset.iloc[:, :-1].values
        y = dataset.iloc[:, -1].values

        if self.oneHotencoding == False:
            return(x,y)
        else:
            return(self.one_hot_encoding(index=3,x=x),y)

    def one_hot_encoding(self,index,x):
        ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[index])],remainder='passthrough')
        X = np.array(ct.fit_transform(x))
        return(X)

    def split(self):
        # Splitting the dataset into trainingSet & testSet
        self.x_train, self.x_test,self.y_train, self.y_test = train_test_split(self.import_Index()[0], self.import_Index()[1], test_size=0.2, random_state=0)
        # return(x_train,x_test,y_train,y_test)

    def train(self):
        self.regressor = LinearRegression()
        self.regressor.fit(self.x_train, self.y_train)

    def prediction(self):
        self.train()
        y_pred = self.regressor.predict(self.x_test)
        return(y_pred)

    def Visualize(self):
        self.split()
        self.train()
        average = []
        total = 0
        plt.scatter(range(len(self.y_test)), self.y_test, color='red')
        plt.scatter(range(len(self.y_test)), self.prediction(), color='blue')
        for i in range(len(self.y_test)):
            if  self.y_test[i]> self.prediction()[i] :
                avg = (self.prediction()[i] / self.y_test[i]) * 100
            else:
                avg = (self.y_test[i] / self.prediction()[i]) * 100
            average.append(avg)
        for i in average:
            total = total + i
        total_percentage = (total/len(average))
        Acc = round(total_percentage,2)
        plt.xlabel('Index')
        plt.ylabel('Profit')
        plt.suptitle('Multiple Regression prediction:Blue. Actual:Red - Average Accuracy: '+str(Acc)+'%')
        plt.show()
mul = Multiple_Regression('50_Startups.csv',oneHotencoding=True)
mul.Visualize()