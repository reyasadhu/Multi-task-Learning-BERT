import csv
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer,  RobertaTokenizer
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def load_sentiment_data(sentiment_filename):
    sentiment_data = []
    with open(sentiment_filename, 'r') as f:
        for row in csv.DictReader(f,delimiter = '\t'):
            sent = row['sentence'].lower().strip()
            sentiment = int(row['sentiment'].strip())
            if sentiment in [0,2,4]: #only keeping the negative, neutral and positive
                if sentiment==2:
                    sentiment=1
                elif sentiment==4:
                    sentiment=2 
            else:
                continue
            
            sentiment_data.append((sent, sentiment))
    return sentiment_data 


def load_negation_data(negation_filename):
    negation_data=[]
                                  
    with open(negation_filename, 'r') as f:
        for row in csv.DictReader(f,delimiter = '\t'):
            sentence1 = row['premise'].lower().strip()
            sentence2= row['hypothesis'].lower().strip()
            negation_data.append((sentence1, sentence2, int(row['label'])))
    return negation_data


def load_sarcasm_data(sarcasm_filename):
    def parseJson(fname):
        for line in open(fname, 'r'):
            data = eval(line)
            yield (data['headline'], data['is_sarcastic'])
    sarcasm_data = list(parseJson(sarcasm_filename))
    return sarcasm_data 
                                 

def load_multitask_data(split='train'):
    
    if split=='train':
        
        sentiment_data = load_sentiment_data('./Dataset/Sentiment SST5/train.csv')
        negation_data = load_negation_data('./Dataset/Cannot_data/train.csv')
        sarcasm_data = load_sarcasm_data('./Dataset/Sarcasm-News-Headline/train.json')
       
        
    elif split=='test':
        
        sentiment_data = load_sentiment_data('./Dataset/Sentiment SST5/test.csv')
        negation_data = load_negation_data('./Dataset/Cannot_data/test.csv')
        sarcasm_data = load_sarcasm_data('./Dataset/Sarcasm-News-Headline/test.json')
       
        
    return sentiment_data, negation_data, sarcasm_data

class PairSentenceDataset(Dataset):

    def __init__(self, dataset, args, isRegression =False):
        self.dataset = dataset
        self.isRegression = isRegression
        if args.transformer=='bert':
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        elif args.transformer=='roberta':
            self.tokenizer =  RobertaTokenizer.from_pretrained('roberta-base')
            
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        
        return self.dataset[idx]

    def pad_data(self, data):
        sent1 = [x[0] for x in data]
        sent2 = [x[1] for x in data]
        labels = [x[2] for x in data]
        
        max_length = int(self.tokenizer.model_max_length//2)-2
        encoding1 = self.tokenizer(sent1, return_tensors='pt', padding='max_length', max_length=max_length, truncation=True)
        encoding2 = self.tokenizer(sent2, return_tensors='pt', padding='max_length', max_length=max_length, truncation=True)

        token_ids = torch.tensor(encoding1['input_ids'], dtype=torch.long)
        attention_mask = torch.tensor(encoding1['attention_mask'], dtype=torch.long)

        token_ids2 = torch.tensor(encoding2['input_ids'], dtype=torch.long)
        attention_mask2 = torch.tensor(encoding2['attention_mask'], dtype=torch.long)
        
        if self.isRegression:
            labels = torch.tensor(labels, dtype=torch.float)
        else:
            labels = torch.tensor(labels, dtype=torch.long)
            

        return (token_ids, attention_mask, token_ids2,attention_mask2, labels)

    def collate_fn(self, all_data):
        (token_ids, attention_mask, token_ids2, attention_mask2, labels) = self.pad_data(all_data)

        batched_data = {
                'token_ids_1': token_ids,
                'attention_mask_1': attention_mask,
                'token_ids_2': token_ids2,
                'attention_mask_2': attention_mask2,
                'labels': labels
            }

        return batched_data 
    
class SingleSentenceDataset(Dataset):

    def __init__(self, dataset, args):
        self.dataset = dataset
        if args.transformer=='bert':
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        elif args.transformer=='roberta':
            self.tokenizer =  RobertaTokenizer.from_pretrained('roberta-base')
            
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def pad_data(self, data):
        sents = [x[0] for x in data]
        labels = [x[1] for x in data]
        max_length = self.tokenizer.model_max_length
        encoding = self.tokenizer(sents, return_tensors='pt', padding='max_length', max_length=max_length, truncation=True)

        token_ids = torch.tensor(encoding['input_ids'], dtype=torch.long)
        attention_mask = torch.tensor(encoding['attention_mask'], dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)

        return (token_ids, attention_mask, labels, sents)

    def collate_fn(self, all_data):
        (token_ids, attention_mask, labels, sents) = self.pad_data(all_data)

        batched_data = {
                'token_ids': token_ids,
                'attention_mask': attention_mask,
                'labels': labels,
                'sents':sents
            }

        return batched_data 

def load_dataloaders(args, config):
    sentiment_train_data, neg_train_data, sar_train_data= load_multitask_data()
    sentiment_test_data, neg_test_data, sar_test_data= load_multitask_data(split ='test')
    
    config.batch_size_individual={}
    
    for task in ['sentiment', 'neg', 'sar']:
        config.gradient_accumulations[task] = int(np.ceil(config.batch_size / config.max_batch_size[task]))
        config.batch_size_individual[task] = config.batch_size // config.gradient_accumulations[task]
        


    # sentiment: Sentiment classification
    sentiment_train_data = SingleSentenceDataset(sentiment_train_data, args)
    sentiment_test_data = SingleSentenceDataset(sentiment_test_data, args)
    sentiment_train_dataloader = DataLoader(sentiment_train_data, shuffle=True, batch_size=config.batch_size_individual['sentiment'],  collate_fn=sentiment_train_data.collate_fn)
    sentiment_test_dataloader = DataLoader(sentiment_test_data, shuffle=False, batch_size=config.batch_size_individual['sentiment'], collate_fn=sentiment_test_data.collate_fn)

    # sar: Sarcasm Classification
    sar_train_data = SingleSentenceDataset(sar_train_data, args)
    sar_test_data = SingleSentenceDataset(sar_test_data, args)
    sar_train_dataloader = DataLoader(sar_train_data, shuffle=True, batch_size=config.batch_size_individual['sar'],  collate_fn=sar_train_data.collate_fn)
    sar_test_dataloader = DataLoader(sar_test_data, shuffle=False, batch_size=config.batch_size_individual['sar'], collate_fn=sar_test_data.collate_fn)
    
    
    # neg: Negation Detection
    neg_train_data = PairSentenceDataset(neg_train_data, args)
    neg_test_data = PairSentenceDataset(neg_test_data, args)
    neg_train_dataloader = DataLoader(neg_train_data, shuffle=True, batch_size=config.batch_size_individual['neg'],
                                        collate_fn=neg_train_data.collate_fn)
    neg_test_dataloader = DataLoader(neg_test_data, shuffle=False, batch_size=config.batch_size_individual['neg'],
                                    collate_fn=neg_test_data.collate_fn)
    
    
    train_dataloaders = {'sentiment': sentiment_train_dataloader, 'sar': sar_train_dataloader, 'neg': neg_train_dataloader}
    
    test_dataloaders = {'sentiment': sentiment_test_dataloader, 'sar': sar_test_dataloader, 'neg': neg_test_dataloader}
    
    return train_dataloaders, test_dataloaders