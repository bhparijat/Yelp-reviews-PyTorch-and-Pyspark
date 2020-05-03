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
from tqdm import tqdm_notebook as tq
import warnings
import pickle as pkl
warnings.filterwarnings("ignore")
import string
import sys
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
#plt.ion()

reviews = pd.read_csv("model/master_reviews_filtered.csv")

ind = random.sample(list(reviews.index),10000)

reviews = reviews[reviews.index.isin(ind)]

print("shape of reviews used for generating pairs is\n", reviews.shape)

#print("shape of master reviews",reviews.shape)

reviews_mod = reviews.dropna()



print("shape of modified master reviews",reviews_mod.shape)

class SentimentAnalysis:
    def __init__(self):
        
        self.map_word = {}
        self.map_index = {}
        self.x_y_pairs = {}
        self.pairs_map = {}
        
    
    def NGRAM(self):
        pass
    
    
    
def tokenize_vocab(map_word,map_index,x):
    
    idx = len(map_word.keys())
    ans = []
    #print(type(x),x)
    for word in x.split():
        if word not in map_word:
            idx+=1
            map_word[word] = idx
            map_index[idx] = word
        ans.append(idx)
            
    return ans


def build_context_center(x,pairs_map,window = 2):
    
    if window not in pairs_map:
        pairs_map[window] = []
        
    #x = x.split()
    for i in range(len(x)):
        
        for j in range(1,window+1):
            
            if (i-j)>0:
                pairs_map[window].append([x[i],x[i-j]])
                
            if (i+j)<len(x):
                pairs_map[window].append([x[i],x[i+j]])
                
sla = SentimentAnalysis()
reviews_mod['tokenized'] = reviews_mod.text.apply(lambda x: tokenize_vocab(sla.map_word,sla.map_index,x))

_ = reviews_mod.iloc[:,].tokenized.apply(lambda x:build_context_center(x,sla.pairs_map))


print("number of pairs_map generated", len(sla.pairs_map[2]))


t =time.time()

with open("model/map.pkl",'wb') as file:
    pkl.dump(sla.pairs_map,file)
    
    
with open("model/map_word.pkl",'wb') as file:
    pkl.dump(sla.map_word,file)
    
with open("model/map_index.pkl",'wb') as file:
        pkl.dump(sla.map_index,file)

print("time taken to save all the pairs information is ", time.time()-t)
    
