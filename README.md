# Kaggle - Covid19 Retweet Prediction

The goal of this project is to accurately predict the number of retweets a tweet will get. The provided dataset is a small subset that was extracted from the COVID19 Twitter dataset that was collect by the DaSciM team during the first wave of lockdowns (March 2020).

## How to make new predictions

- make_predictions.ipynb: a Jupyter notebook to test our model on new datasets.
    - Insert your new .csv file in /data folder;
    - Change #path in pd.read_csv(#path);
    - Run all the cells and the results will be dumped into "predictions_new.txt" file. 


## Files description

- pipeline.ipynb: a Jupyter notebook that gathers all the steps of our work.

- feature_extraction.py, classification.py, regression.py, prediction.py : Python scripts containing classes and functions relative to their title.

- data:
    - Create a data folder. 
    - You should insert here train.csv and evaluation.csv files that can be found at https://www.kaggle.com/c/covid19-retweet-prediction-challenge-2020/data?select=data
    - If you run the whole pipeline.ipynb file, you will obtain:
        - train_features.csv: a CSV file with precomputed features for train.csv.
        - eval_features.csv: a CSV file with precomputed features for evaluation.csv.

- saved_models: a folder containg our models.

- FeatureExtraction_Text_train: a FeatureExtraction_Text object to use transform function on new datasets. 

- hashtags_dict: a dictionary whose keys are all the unique hastags found in the training set, and whose values are the sum of retweets every hashtag has gotten. 
