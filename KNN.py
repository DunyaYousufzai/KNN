import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
#!pip install scikit-plot
import sklearn.metrics as metrics
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve

class KNN():
    def __init__(self, file):
        self.file = file
    def data(self, x,y,z, ts):
        global data, real_x, real_y, training_x, training_y,testing_x,testing_y
        data = pd.read_csv(self.file)
        real_x = data.iloc[:,x:y].values
        real_y = data.iloc[:,z].values
        training_x, testing_x,training_y,testing_y = train_test_split(real_x, real_y,test_size = ts, random_state = 0)
    
    def filter_dataset(self):
        global scaler, training_x, testing_x
        # convert data between 2 and -2
        scaler = StandardScaler()
        training_x = scaler.fit_transform(training_x)
        testing_x = scaler.fit_transform(testing_x)

    def training(self, n_neighbors, metric):
        global cls
        cls = KNeighborsClassifier(n_neighbors = n_neighbors, metric = metric, p =2)
        cls.fit(training_x, training_y)

    def predict(self):
        global y_pred
        y_pred = cls.predict(testing_x)
        #print actual and predicted values in a table
        for x in range(len(y_pred)):
            print('Actual ',testing_y[x],' Predicted ',y_pred[x])
            if y_pred[x]>=0.5:
                y_pred[x]=1
            else:
                y_pred[x]=0
    def confusionMatrix(self):
        global confusion
        # first with forth are correct predictions and two with three are wrong predictons
        confusion = confusion_matrix(testing_y, y_pred)
        print(confusion)
    
    def training_plot(self, title, xlabel, ylabel):
        global X_set, y_set, X1,X2
        #  -1 means minimum data (-2)
        # + 1 means maximum data (+2)
        # ravel helps for mid points
        X_set, y_set = training_x, training_y
        X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                            np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
        # draw the middle line
        plt.contourf(X1, X2, cls.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                    alpha = 0.75, cmap = ListedColormap(('green', 'black')))
        plt.xlim(X1.min(), X1.max())
        plt.ylim(X2.min(), X2.max())

        for i, j in enumerate(np.unique(y_set)):
           plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('green', 'black'))(i), label = j)
            
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.show()

    def testing_plot(self, title, xlabel, ylabel):
        global X_set, y_set, X1,X2
        #  -1 means minimum data (-2)
        # + 1 means maximum data (+2)
        # ravel helps for mid points
        X_set, y_set = testing_x, testing_y
        X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                            np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
        # draw the middle line
        plt.contourf(X1, X2, cls.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                    alpha = 0.75, cmap = ListedColormap(('brown', 'black')))
        plt.xlim(X1.min(), X1.max())
        plt.ylim(X2.min(), X2.max())

        for i, j in enumerate(np.unique(y_set)):
           plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('brown', 'black'))(i), label = j)
            
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.show()

    def auc(self):
        global y_pred_proba
        y_pred_proba = cls.predict_proba(testing_x)[::,1]
        fpr, tpr, _ = metrics.roc_curve(testing_y,  y_pred_proba)
        auc = metrics.roc_auc_score(testing_y, y_pred_proba)
        plt.plot(fpr,tpr,label="data , auc="+str(auc))
        plt.legend(loc=4)
        plt.show()

    def plot_roc_curve(self, fpr, tpr):
        plt.plot(fpr, tpr, color='blue', label='ROC')
        plt.plot([0, 1], [0, 1], color='purple', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend()
        plt.show()
    
    def average_precision(self):
        global average_precision
        average_precision = average_precision_score(testing_y, y_pred_proba)
        print('Average precision-recall score: {0:0.2f}'.format(
            average_precision))
    
    def precision_recall_curve(self):
        disp = plot_precision_recall_curve(cls, testing_x, testing_y)
        disp.ax_.set_title('2-class Precision-Recall curve: ' 'AP={0:0.2f}'.format(average_precision))
        plt.show()


lg = KNN("KNN.csv")
lg.data(2,4,4, 0.25)
lg.filter_dataset()
lg.training(5 , 'minkowski')
lg.predict()
lg.confusionMatrix()
lg.training_plot('Logistic Regression (Training set)','Age','Estimated Salary')
lg.testing_plot('Logistic Regression (Testing set)','Age','Estimated Salary')
lg.auc()
fpr, tpr, thresholds = metrics.roc_curve(testing_y, y_pred_proba)
lg.plot_roc_curve(fpr, tpr)
lg.average_precision()
lg.precision_recall_curve()
