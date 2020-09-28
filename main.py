import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from watson_developer_cloud import NaturalLanguageUnderstandingV1
from watson_developer_cloud.natural_language_understanding_v1 import Features, EntitiesOptions, KeywordsOptions, SentimentOptions, CategoriesOptions
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
import re
nltk.download('stopwords')
import sys
from itertools import repeat
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from wordcloud import WordCloud
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import seaborn as sns

import numpy as np


def readDf():
    analyzer = SentimentIntensityAnalyzer()
    df = pd.read_csv('Arthur-Data-Sept-EU.csv')

    ### Map the language code with country
    langD = {'Italia': 'it', 'Deutschland': 'de', 'France': 'fr-FR', 'España': 'es-ES'}
    df['langCode'] = df['review_country'].map(langD)

    return df

def IBMtran(df):
    natural_language_understanding = NaturalLanguageUnderstandingV1(
        version='2019-07-12',
        iam_apikey='v8j7M76fx4hOFr35AhLUso35qgmsocV5_WM-Ag0IdKg5',
        url='https://api.us-south.natural-language-understanding.watson.cloud.ibm.com/instances/dbf791a6-366c-48d9-81ac-9a08ac7f130c')

    ibm = []
    i = 0
    x = 0.5

    def senti(x):
        ### Vader Sentiment
        analyzer = SentimentIntensityAnalyzer()
        try:
            response = natural_language_understanding.analyze(
                text=str(x),
                features=Features(sentiment=SentimentOptions())).get_result()
            res = response.get('sentiment').get('document').get('score')
            #         response1 = natural_language_understanding.analyze(
            #                text = str(df['review_title'][i]),
            #                features = Features(sentiment=SentimentOptions())).get_result()
            #         res1 = response1.get('sentiment').get('document').get('score')
            #         final = (res*x + res1*(1-x))
            return res
        except:
            vs = analyzer.polarity_scores(str(x))
            return vs['compound']

    df['ibm2'] = df.apply(lambda x: senti(x['review_body_english']), axis=1)
    return df

def IBMNonTran(df):
    natural_language_understanding = NaturalLanguageUnderstandingV1(
        version='2019-07-12',
        iam_apikey='v8j7M76fx4hOFr35AhLUso35qgmsocV5_WM-Ag0IdKg5',
        url='https://api.us-south.natural-language-understanding.watson.cloud.ibm.com/instances/dbf791a6-366c-48d9-81ac-9a08ac7f130c')

    ibm = []
    i = 0
    x = 0.5

    for i in range(0, len(df)):
        try:
            response = natural_language_understanding.analyze(language=df['langCode'][i],
                                                              text=str(df['review_body'][i]),
                                                              features=Features(
                                                                  sentiment=SentimentOptions())).get_result()
            res = response.get('sentiment').get('document').get('score')

            ibm.append(res)
        except:
            ibm.append('NA')

    df['ibm1'] = ibm
    return df


def main():
    df = readDf()
    IBMop = IBMNonTran(df[1:5])

if __name__ == "__main__":
    main()