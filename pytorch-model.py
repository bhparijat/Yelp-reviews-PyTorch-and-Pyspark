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
    
    
    

class TrainSelfEmbedding:
    
    def __init__(self):
        self.window_size = 2
        self.epochs = 100
        self.embeddings = 100
        self.epoch_loss = 0
        
    def train(self,pairs_map,epochs = 100,sample_size=10**4):
        
        self.vocab_size = len(sla.map_word.keys())
        
        print(self.vocab_size)
        
        
        weight1 = torch.randn(self.embeddings,self.vocab_size).float()
        weight2 = torch.randn(self.vocab_size,self.embeddings).float()
        
        print(sys.getsizeof(weight1.storage()),weight1.element_size()*weight1.nelement()/10**6)
        print(sys.getsizeof(weight2.storage()),weight2.element_size()*weight2.nelement()/10**6)
        
        #total_loss = 0
        
        self.epochs = epochs
        
        loss_per_epoch = []
        for i in tq(range(self.epochs)):
            
            total_loss = 0
            
            
            sampled_pairs = random.sample(pairs_map[self.window_size],sample_size)
            
            for i in tq(range(sample_size)):

                center,target = sampled_pairs[i]
                inpt = torch.zeros(self.vocab_size).float()
                inpt[center] = 1.0
                z1 = torch.matmul(weight1,inpt)
                a1 = torch.relu(z1)
                
                #print(a1.size(),z1.size())
                
                z2 = torch.matmul(weight2,z1)

                log_output = torch.log_softmax(z2,dim=0)

#                 print(inpt.size())
#                 print(weight1.size())
#                 print(z1.size())


#                 print(weight2.size())
#                 print(z2.size())
#                 print(log_output.size())

                loss = torch.nn.functional.nll_loss(log_output.view(1,-1),torch.tensor([target]))

                total_loss += loss
                
                print(loss)

            loss_per_epoch.append(total_loss/sample_size)
            
            print(total_loss)
            return
            
        return loss_per_epoch
            
        
        
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
    
    
self_embedding = TrainSelfEmbedding()
    
loss = self_embedding.train(sla.pairs_map,epochs= 50, sample_size= 1000000)
