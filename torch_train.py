import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
import torchvision
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader,Dataset, random_split
import matplotlib.pyplot as plt
import random 
import time
from tqdm import tqdm_notebook as tq
import warnings
import pickle as pkl
warnings.filterwarnings("ignore")
import string
import sys
from nltk.corpus import stopwords
from torch.autograd import Variable
plt.ion()
BATCH_SIZE = 32
epochs = 10
num_class = 2
embed_dim = 100
device = torch.device("gpu" if torch.cuda.is_available() else "cpu")
class YelpReviewsSentimentAnalysis(nn.Module):
    
    def __init__(self,vocab_size,embed_dim,num_class):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()
        
    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()
        
    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)
class YelpDataset(Dataset):
    
    def __init__(self,json_file,threshold=3):
        
        self.raw_data = pd.read_json(json_file,lines=True)
    
        self.raw_data['label'] = self.raw_data.stars.apply(lambda x : 1 if x>=3 else 0)
        
        self.raw_data = self.raw_data[["label","text"]].iloc[:100000,]
        
        self.word2idx = {}
        
        self.idx2word = {}
        
        self.word2freq = {}
        self.word_count = 0
        
        self.maxLen = 0
        self.__init__preprocess()
        
        self.data = self.raw_data.to_numpy()
   
        
       
        
    def __len__(self):
        return self.data.shape[0]
    
    
    def __getitem__(self,idx):
        
        
        sample = self.data[idx,:]
            
        return sample
    
    def __init__preprocess(self):
        
        
        def clean(text):
            text = text.lower()
        
            text = [ch for ch in text if ch not in string.punctuation]


            text = "".join(text)

            text = [c for c in text if c == " " or c.isalnum()]

            text = "".join(text)

            stop_words = set(stopwords.words("english"))

            text = text.split(" ")

            text = [word for word in text if word not in stop_words]
        
        
            text = " ".join(text)
            
            return text
          
            
        def build_vocab(text):
            
            
            text = text.split(" ")
            
            text_token = []
            
            for word in text:
                
                if word not in self.word2idx:
                    
                    self.word2idx[word] = self.word_count
                    
                    self.idx2word[self.word_count] = word
                    
                    self.word_count+=1
                    
                    
                text_token.append(self.word2idx[word])
             
            self.maxLen = max(self.maxLen,len(text_token))
            return text_token
        
        self.raw_data['text'] = self.raw_data.text.apply(lambda x : clean(x))
        self.raw_data['text'] = self.raw_data.text.apply(lambda x : build_vocab(x))
def collate_offset(batch):
    
    text = [torch.tensor(x[1]) for x in batch]
    
    label = torch.tensor([x[0] for x in batch])
    offsets = [0] + [len(x) for x in text]
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    
    text = torch.cat(text)
    
    return text, offsets, label
def train_func(train):
    
    train_data = DataLoader(train,batch_size=BATCH_SIZE,shuffle=True,collate_fn=collate_offset,num_workers=4)
    
    train_loss = 0
    train_acc = 0
    for i, (text, offsets, label) in enumerate(train_data):
        optimizer.zero_grad()
        
        text, offsets, label = text.to(device), offsets.to(device), label.to(device)
        
    
        output = model(text, offsets)
        
        loss = criterion(output, label)
        
        #loss = Variable(loss,requires_grad=True)
        
        train_loss += loss.item()
        
        loss.backward()
        optimizer.step()
        
        train_acc += (output.argmax(1) == label).sum().item()

    
    scheduler.step()
    
    return train_loss / len(train), train_acc / len(train)
        
def test_func(test):
    
    test_data = DataLoader(test,batch_size=BATCH_SIZE,shuffle=True,collate_fn=collate_offset,num_workers=4)
    
    test_loss = 0
    test_acc = 0
    
    for i, (text, offsets, label) in enumerate(test_data):
        
        text, offsets, label = text.to(device), offsets.to(device), label.to(device)

        with torch.no_grad():
            
            output = model(text, offsets)

            loss = criterion(output, label)
            
            #loss = Variable(loss,requires_grad=True)
            
            test_loss += loss.item()
            
           
            test_acc += (output.argmax(1) == label).sum().item()   
    
    return test_loss / len(test), test_acc / len(test)
if __name__ == "__main__":
    
    
    
    start_time = time.time()
    yelp_dataset = YelpDataset(json_file = "~/data/yelp/review.json")
    
    print("Time Taken to for dataset preprocessing is {} minutes...".format((time.time()-start_time)/60))
    
    
    train_len = int(len(yelp_dataset)*0.8)
    
    valid_len = int(len(yelp_dataset)*0.1)
    
    test_len = len(yelp_dataset) - train_len -valid_len
    
    train,valid,test = random_split(yelp_dataset,[train_len,valid_len,test_len])
    
    
    
    print("Length of training dataset is {}".format(train_len))
          
    print("Length of validation dataset is {}".format(valid_len))
          
      
    model = YelpReviewsSentimentAnalysis(yelp_dataset.word_count,embed_dim, num_class)
    
    
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=4.0)
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)
     
    tl_,ta_ = [],[]
    vl_,va_ = [],[]
    
    for epch in range(epochs):
          
        start_time = time.time()
        train_loss, train_acc = train_func(train)
        valid_loss, valid_acc = test_func(valid)

        tl_.append(train_loss)
        ta_.append(train_acc)
        
        
        vl_.append(valid_loss)
        va_.append(valid_acc)
        
        
        secs = int(time.time() - start_time)
        mins = secs / 60
        
        
        
        name = "model_"+str(epch)+".pkl"
        with open(name,"wb") as file:
            pkl.dump(model,file)
            
        print("{} saved..".format(name))
        print("Time Taken to complete epoch is {} minutes...".format(mins))
