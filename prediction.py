import numpy as np
import pandas as pd
import pickle 
import csv

from feature_extraction import FeatureExtraction, FeatureExtraction_Text


def transform_eval(eval_df):
    #Feature extraction for a new dataset. 
    
    FE_text = pickle.load(open('FeatureExtraction_Text_train','rb'))
    FE_eval = FeatureExtraction(eval_df)
    FE_eval.transform()
    FE_text_eval = FE_text
    FE_text_eval.transform(df=eval_df)
    features_df = pd.concat([FE_eval.transformed_df, FE_text_eval.transformed_df], axis=1)
    return features_df


def pred_eval(classifier, regressor, eval_features):
    
    eval_df_classifier = classifier.sc_class.fit_transform(eval_features.copy())
    y_pred = pd.DataFrame(np.zeros(len(eval_features)), columns = ["pred"])
    
    #classification
    y_test_class_pred = classifier.model.predict(eval_df_classifier)
 
    #spliting based on user_followers_count
    eval_df_concat = eval_features.copy()
    eval_df_concat["class"] = eval_df_concat["user_followers_count"].apply(lambda x: regressor.label_followers_count(x, regressor.bins))

    eval_df_reg = eval_df_concat.iloc[:,:-1]
    eval_df_class = eval_df_concat.iloc[:,-1]
    eval_df_class.loc[y_test_class_pred==0] = 0

    #regression
    bins = regressor.bins
    target_names = ["0"]
    target_names.extend(["≤ " + str(i) for i in bins])
    target_names.append("+"+str(bins[-1]))

    for c, elm in enumerate(target_names):

        X_ = eval_df_reg.loc[eval_df_class==c]

        if c!=0:
            X_ = regressor.scalers[c-1].transform(X_)
            y_ = np.abs(regressor.regressors[c-1].predict(X_)[:,np.newaxis])
        else:
            y_ = np.zeros((len(X_),1))

        y_pred.loc[eval_df_class==c] = y_
    
    return y_pred


def save_pred(eval_df, y_pred, filename="predictions.txt"):
    with open(filename, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["TweetID", "NoRetweets"])
        for index, prediction in enumerate(y_pred.values):
            writer.writerow([str(eval_df['id'].iloc[index]) , str(prediction[0])])


