import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
import torchvision
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random 
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
import string
from nltk.corpus import stopwords

def clean_data(reviews):
    reviews.text = reviews.text.apply(lambda x: x.lower())
    
    def remove_punctuation(s):
    #s = s.split()
    
        s = [ch for ch in s if ch not in string.punctuation]

        
        s = "".join(s)
        
        s = [c for c in s if c == " " or c.isalnum()]
        
        s = "".join(s)
        
        return s
    
    
    def remove_stopwords(s):
        stop_words = set(stopwords.words("english"))

        s = s.split(" ")

        s = [word for word in s if word not in stop_words]
        
        
        s = " ".join(s)
 
        
        return s
    
    
    reviews.text = reviews.text.apply(lambda x:remove_punctuation(x))
    reviews.text = reviews.text.apply(lambda x:remove_stopwords(x))
    
    return reviews

reviews = pd.read_csv("model/reviews_filtered.csv",lines=True)
reviews_cleaned = reviews.copy()
reviews_cleaned = clean_data(reviews_cleaned)
filter_null_row_cond = reviews_cleaned.text == "null"
reviews_cleaned = reviews_cleaned[~filter_null_row_cond]

reviews['text_length'] = reviews.text.apply(lambda x:len(x))

print("columns\n",reviews_cleaned.columns)
reviews_cleaned['label']= reviews.stars.apply(lambda x: 1 if x>=3 else 0)

print("shape\n", reviews.shape)
reviews_cleaned.to_csv("model/master_reviews_filtered.csv",index=False)