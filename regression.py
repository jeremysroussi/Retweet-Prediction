import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import LinearSVR

from sklearn.metrics import mean_absolute_error

from tqdm.notebook import tqdm
tqdm.pandas()


class Regression():
    
    def __init__(self, bins):
        self.bins=bins
    
    def label_followers_count(self, count, bins):
        """Assign a class to a number of followers.

        Args:
            count (int): number of followers
            bins (list): list of thresholds
        Returns:
            int: class number
        """
        for i, elm in enumerate(bins):
            if count <= elm:
                return i+1
        return i+2
        
    def regression_per_class(self, X_train, y_train, features):
    #Compute a regressor for every class of followers.
        
        X_train_reg = X_train.copy()
        X_train_reg['class'] = X_train_reg["user_followers_count"].apply(lambda x: self.label_followers_count(x, self.bins))
        X_train_reg['class'][y_train==0] = 0

        regressors=[]
        scalers=[]

        X_train_concat = pd.concat([X_train_reg, y_train], axis=1)
        by_class = X_train_concat.loc[X_train_concat["class"]!=0].groupby(["class"])

        for _, X_y_concat in tqdm(by_class):

            X_train_r = X_y_concat.iloc[:,:-2]
            y_train_r = X_y_concat.iloc[:,-1]

            sc = RobustScaler()
            X_train_r = sc.fit_transform(X_train_r)
            
            #model = LinearSVR()

            params = {'n_estimators': 50,
                    'max_depth': 5,
                    'min_samples_split': 5,
                    'learning_rate': 0.1,
                    'loss': 'lad'}  #least absolute deviation
            model = GradientBoostingRegressor(**params)
            model.fit(X_train_r, y_train_r)

            regressors += [model]
            scalers+=[sc]
            
        self.regressors = regressors  
        self.scalers = scalers 


def model_performance(classifier, regressor, features, X_test, y_test):
    
    X_test_classifier = classifier.sc_class.fit_transform(X_test.copy())
    y_pred = pd.DataFrame(np.zeros(len(X_test)), columns = ["pred"])
    
    #classification
    y_test_class_pred = classifier.model.predict(X_test_classifier)
 
    #spliting based on user_followers_count
    X_test_concat = X_test.copy()
    X_test_concat["class"] = X_test_concat["user_followers_count"].apply(lambda x: regressor.label_followers_count(x, regressor.bins))

    X_test_reg = X_test_concat.iloc[:,:-1]
    X_test_class = X_test_concat.iloc[:,-1]
    X_test_class.loc[y_test_class_pred==0] = 0

    #regression
    bins = regressor.bins
    target_names = ["0"]
    target_names.extend(["≤ " + str(i) for i in bins])
    target_names.append("+"+str(bins[-1]))

    print("Prediction error (MAE):")
    for c, elm in enumerate(target_names):

        X_ = X_test_reg.loc[X_test_class==c]

        if X_.shape[0]!=0:
            if c!=0:
                X_ = regressor.scalers[c-1].transform(X_)
                y_ = np.abs(regressor.regressors[c-1].predict(X_)[:,np.newaxis])
            else:
                y_ = np.zeros((len(X_),1))

            y_pred.loc[X_test_class==c] = y_
            print(target_names[c], ":", 
            round(mean_absolute_error(y_true=y_test.loc[X_test_class==c], y_pred=y_),2,), "-", X_.shape[0])

            if c==0:
                print("For non zero prediction, regression based on user_followers_count:")

    print("Overall:", mean_absolute_error(y_true=y_test, y_pred=y_pred))