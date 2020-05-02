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



with open("model/map.pkl","rb") as file:
    map_pairs = pkl.load(file)

with open("model/map_word.pkl", "rb") as file:
    map_word = pkl.load(file)


class TrainSelfEmbedding:
    
    def __init__(self):
        self.window_size = 2
        self.epochs = 100
        self.embeddings = 100
        self.epoch_loss = 0
        
    def train(self,pairs_map,epochs = 100,sample_size=10**6):
        
        self.vocab_size = len(map_word.keys())
        
        print("vocab size",self.vocab_size)
        
        
        weight1 = torch.randn(self.embeddings,self.vocab_size).float()
        weight2 = torch.randn(self.vocab_size,self.embeddings).float()
        
        print(sys.getsizeof(weight1.storage()),weight1.element_size()*weight1.nelement()/10**6)
        print(sys.getsizeof(weight2.storage()),weight2.element_size()*weight2.nelement()/10**6)
        
        #total_loss = 0
        
        self.epochs = epochs
        
        loss_per_epoch = []
        for i in range(self.epochs):
            
            total_loss = 0
            
            
            sampled_pairs = random.sample(pairs_map[self.window_size],sample_size)
            t = time.time()
            for i in range(sample_size):

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
                
                
            loss_per_epoch.append(total_loss/sample_size)
            
            print("Total loss {} for epoch {} in time {}".format(total_loss,i+1,time.time()-t ))
            
            
        return loss_per_epoch,weights1,weights2

self_embedding = TrainSelfEmbedding()

a = time.time()
loss,w1,w2 = self_embedding.train(sla.pairs_map,epochs= 50, sample_size= 1000000)
print("Training done in time ",time.time()-a)

with open("model/w1.pkl","wb") as file:

    pkl.dump(w1,file)

with open("model/w2.pkl" "wb") as file:
    pkl.dump(w2,file)
