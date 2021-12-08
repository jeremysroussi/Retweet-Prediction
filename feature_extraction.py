import pandas as pd
import numpy as np
import pickle

from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import time
from datetime import datetime

from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

from tqdm.notebook import tqdm
tqdm.pandas()
    

class FeatureExtraction():
    
    def __init__(self, df):
        self.df = df
    
    def score_hashtags(self, x, hashtags_dict):
        #Gives a score to every hashtag. 
        if x == 0:
            return 0
        else:
            ht = x.split(", ")
            s = [hashtags_dict[h] if h in hashtags_dict.keys() else 0 for h in ht]
        return np.sum(s)
        
    def transform(self):
        df = self.df
        #sentiment
        '''
        sentiment = pd.Series(df['text']).progress_apply(lambda x: TextBlob(x).sentiment)
        polarity = sentiment.apply(lambda x: x[0])
        subjectivity = sentiment.apply(lambda x: x[1])
        df['polarity']=polarity
        df['subjectivity']=subjectivity
        ''' 
        #sentiment analysis with VADER 
        analyser = SentimentIntensityAnalyzer()
        score = pd.Series(df['text']).progress_apply(lambda x: analyser.polarity_scores(x))
        df['positive'] = score.apply(lambda x: x['neg'])
        df['neutral'] = score.apply(lambda x: x['neu'])
        df['negative'] = score.apply(lambda x: x['pos'])
        df['compound'] = score.apply(lambda x: x['compound'])

        #timestamp
        date = pd.to_datetime(df['timestamp'], unit='ms')
        df['hour'] = date.dt.hour
        df['minute'] = date.dt.minute

        #verified
        df["user_verified"]=df["user_verified"].astype(int)

        #hashtags
        df["hashtags"].replace(np.nan, "", inplace = True)
        df["num_hashtags"]=df["hashtags"].apply(lambda x : len(x.split(", ")) if x!= "" else 0)
        df['text']=df['text'].apply(lambda x: x.replace('\r',''))
        #hashtag scores 
        hashtags_dict = pickle.load(open("hashtags_dict",'rb'))
        df["hashtag_score"]=df["hashtags"].apply(lambda x: self.score_hashtags(x,hashtags_dict))
        df["hashtag_score"] = df["hashtag_score"]/df["hashtag_score"].max()

        #length
        df["length"]=df["text"].apply(lambda x : len(TextBlob(x).split(" ")))
        
        #num_mentions
        df["user_mentions"].replace(np.nan, "", inplace = True)
        df["num_mentions"]=df["user_mentions"].apply(lambda x : len(x.split(", ")) if x!= "" else 0)
        
        #num_urls
        df["urls"].replace(np.nan, "", inplace = True)
        df["num_urls"]=df["urls"].apply(lambda x : len(x.split(", ")) if x!= "" else 0)
        
        #fillna
        df.fillna(0,inplace = True)
        
        self.transformed_df = df
        
        pass
    
class FeatureExtraction_Text():
    #Dimension reduction on the most common words using PCA.  

    def __init__(self, df, max_features=100, dim_pca=20):
       

        self.df = df
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')

        self.dim_pca = dim_pca
        self.pca = PCA(n_components=dim_pca)
    
    def fit(self):
        df = self.df
        X_text = self.vectorizer.fit_transform(df['text'])
        df_text = pd.DataFrame(X_text.toarray(), columns=self.vectorizer.get_feature_names())
        self.pca.fit(df_text)

    def transform(self, df):
        X_text = self.vectorizer.transform(df['text'])
        features_reduced = self.pca.transform(X_text.toarray())
        features_text = pd.DataFrame(features_reduced, 
                                        columns=['PCA'+str(i) for i in range(1,self.dim_pca+1)])

        self.transformed_df = features_text
        
        pass