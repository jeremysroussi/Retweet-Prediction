import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import classification_report
from sklearn.metrics import mean_absolute_error

class Classification():
    
    def __init__(self, bins=[0]):
        self.sc_class = RobustScaler()
        self.bins = bins
                
                
    def classify(self, X_train, y_train_class):      

        X_train_classifier = self.sc_class.fit_transform(X_train.copy())

        #some classifier
        rf = RandomForestClassifier(criterion= 'gini', max_depth= 17, max_features= 4, n_estimators=14)
        rf.fit(X_train_classifier, y_train_class)
        
        self.model = rf
        
        pass


def classifier_performance(classifier, X_test, y_test_class):
    X_test_classifier = classifier.sc_class.fit_transform(X_test.copy())
    y_pred_class = classifier.model.predict(X_test_classifier)

    bins = classifier.bins
    target_names = ["0"]
    target_names.extend(["≤ " + str(i) for i in bins[1:]])
    target_names.append("+"+str(bins[-1]))
    print(classification_report(y_test_class, y_pred_class, target_names=target_names))

