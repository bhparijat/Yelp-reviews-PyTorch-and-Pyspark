import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
import torchvision
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import random
import time
#from tqdm import tqdm_notebook as tq
import warnings
import pickle as pkl
warnings.filterwarnings("ignore")
import string
import sys
import nltk
#nltk.download('stopwords')                                                                                                                                                                               
from nltk.corpus import stopwords
#plt.ion()      


t = time.time()
print("loading pairs map******************")
with open("model/map.pkl","rb") as file:
    pairs_map = pkl.load(file)

print("loading map word******************")
with open("model/map_word.pkl", "rb") as file:
    map_word = pkl.load(file)

print("time taken to load pickle files ", time.time()-t)

class TrainSelfEmbedding:
    
    def __init__(self):
        self.window_size = 2
        self.epochs = 100
        self.embeddings = 100
        self.epoch_loss = 0
        
    def train(self,pairs_map,epochs = 100,sample_size=10**6,lr=0.001):
        
        print("initializing vocab size")
        self.vocab_size = len(map_word.keys())+1
        
        print("vocab size",self.vocab_size)
        
        
        weight1 = torch.randn(self.embeddings,self.vocab_size,requires_grad=True).float()
        weight2 = torch.randn(self.vocab_size,self.embeddings,requires_grad=True).float()
        
        print(sys.getsizeof(weight1.storage()),weight1.element_size()*weight1.nelement()/10**6)
        print(sys.getsizeof(weight2.storage()),weight2.element_size()*weight2.nelement()/10**6)
        
        #total_loss = 0
        
        self.epochs = epochs
        
        sampled_pairs = random.sample(pairs_map[self.window_size],sample_size);
        loss_per_epoch = []
        for ep in range(self.epochs):
            
            total_loss = 0
            
            
            #sampled_pairs = random.sample(pairs_map[self.window_size],sample_size)
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
                loss.backward()
                
                weight1.data -= lr* weight1.grad.data
                weight2.data -= lr * weight2.grad.data

                weight1.grad.data.zero_()
                
                weight2.grad.data.zero_()
            loss_per_epoch.append(total_loss/sample_size)
            
            print("Total loss {} for epoch {} in time {}".format(total_loss,ep+1,time.time()-t ))
            
            
        return loss_per_epoch,weight1,weight2

self_embedding = TrainSelfEmbedding()

a = time.time()

print("calling train function")

loss,w1,w2 = self_embedding.train(pairs_map,epochs= 100, sample_size= 100000,lr=0.001)
print("Training done in time ",time.time()-a)

with open("model/w1.pkl","wb") as file:

    pkl.dump(w1,file)

with open("model/w2.pkl","wb") as file:
    pkl.dump(w2,file)
