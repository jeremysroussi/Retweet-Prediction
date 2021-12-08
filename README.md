# Retweet-Prediction

The goal of this project is to accurately predict the number of retweets a tweet will get. The provided dataset is a small subset that was extracted from the COVID19 Twitter dataset that was collect by the DaSciM team during the first wave of lockdowns (March 2020).

## Project Structure 

.
├── README.md
├── make_predictions.ipynb
├── pipeline.ipynb
├── feature_extraction.py
├── classification.py
├── regression.py
├── prediction.py
├── data
│   ├── train.csv
│   ├── evaluation.csv
│   ├── train_features.csv
│   └── eval_features.csv
├── saved_models
│   ├── classification
│   └── regression
├── predictions.txt
├── FeatureExtraction_Text_train
└── hashtags_dict


## How to make new predictions

• make_predictions.ipynb: a Jupyter notebook to test our model on new datasets.
    - Insert your new .csv file in /data folder;
    - Change #path in pd.read_csv(#path);
    - Run all the cells and the results will be dumped into "predictions_new.txt" file. 


## File description

• pipeline.ipynb: a Jupyter notebook that gathers all the steps of our work.

• feature_extraction.py, classification.py, regression.py, prediction.py : Python scripts containing classes and functions relative to their title.

• data: 
    - You should insert here train.csv and evaluation.csv files.
    - If you run the whole pipeline.ipynb file, you will obtain: 
        - train_features.csv: a CSV file with precomputed features for train.csv. 
        - eval_features.csv: a CSV file with precomputed features for evaluation.csv. 

• saved_models: a folder containg our models.

• FeatureExtraction_Text_train: a FeatureExtraction_Text object to use transform function on new datasets. 

• hashtags_dict: a dictionary whose keys are all the unique hastags found in the training set, and whose values are the sum of retweets every hashtag has gotten. 
