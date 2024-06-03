import os, csv, sys, copy, random
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from contextlib import nullcontext
from tqdm import tqdm
from itertools import cycle
from reproduce_results import reproduce_results
from save_model import save_model
from model import MultiTaskSentimentAnalysis
from scheduler import RoundRobinScheduler, RandomScheduler, ObjectsGroup
from evaluation import model_eval_individual, model_eval_multitask
from transformers import BertModel, RobertaModel, BertTokenizer
from dataset import load_dataloaders
import warnings
warnings.filterwarnings("ignore")

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
class SentimentAnalysisModel(nn.Module):
    def __init__(self, model_name, num_labels):
        super(SentimentAnalysisModel, self).__init__()
        if model_name=='bert-base-uncased':
            self.bert = BertModel.from_pretrained(model_name)
        elif model_name=='roberta-base':
            self.bert = RobertaModel.from_pretrained(model_name)
        self.drop = nn.Dropout(p=0.3)
        self.classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(self.bert.config.hidden_size, num_labels)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1] 
#         output = self.drop(pooled_output)
#         logits = self.classifier(output)
#         return logits
        return pooled_output
    
    def last_layers_sentiment(self, x):

#         for i in range(len(self.linear_sentiment) - 1):
#             x = self.dropout_sentiment[i](x)
#             x = self.linear_sentiment[i](x)
#             x = F.relu(x)
            
#         x = self.dropout_sentiment[-1](x)
#         logits = self.linear_sentiment[-1](x)
        x = self.drop(x)
        logits = self.classifier(x)
        return logits
    
def train_(model, optimizer, train_loader, criterion, device):
    model.train()
    total_loss = 0
    total_acc = 0
    num_samples= 0
    for batch in tqdm(train_loader):
        optimizer.zero_grad()
        
        input_ids = batch['token_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        predictions = torch.argmax(outputs, dim=1)
        total_acc += (predictions == labels).sum().item()
        num_samples += labels.size(0)
        loss = criterion(outputs, labels)/16
        print(loss)
        loss.backward()
        optimizer.step()
#         for name, param in model.named_parameters():
#             print(f"Parameter: {name}")
#             print(f"Gradient:\n{param.grad}")
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        total_loss += loss.item()
    
    print(f'Training loss: {total_loss/len(train_loader)}, Training Acc:{total_acc/num_samples*100}%' )
def load_sentiment_data(sentiment_filename):
    sentiment_data = pd.DataFrame(columns=['id','sentence','sentiment'])
    rows=[]
    with open(sentiment_filename, 'r') as fp:
        for record in csv.DictReader(fp,delimiter = '\t'):
            sent = record['sentence'].lower().strip()
            sent_id = record['id'].lower().strip()
            sentiment = int(record['sentiment'].strip())
            if sentiment==1:
                sentiment=0 # both negative and very negative is mapped to negative
            elif sentiment==2:
                sentiment=1
            elif sentiment ==3 or sentiment==4:
                sentiment=2
            
            rows.append({'id': sent_id, 'sentence': sent, 'sentiment': sentiment})
        sentiment_data = pd.concat([sentiment_data, pd.DataFrame(rows)], ignore_index=True)
        sentiment_data['sentiment']=sentiment_data['sentiment'].astype('int')
        return sentiment_data 
    
class LoadDatasetLLM(Dataset):
    def __init__(self, tokens, labels):
        self.input_ids = tokens['input_ids']
        self.attention_mask = tokens['attention_mask']
        self.labels = torch.tensor(labels.values)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'token_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx]
        }
def check(device, train_loader):
#     # to_delete
#     current_dir = os.path.dirname(__file__)
#     current_dir = os.path.join(current_dir, '..', "Dataset")
#     sentiment_train_data=load_sentiment_data(os.path.join(current_dir,"Sentiment SST5/train.csv"))
#     sentiment_test_data=load_sentiment_data(os.path.join(current_dir,"Sentiment SST5/test.csv"))
#     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

#     max_length = tokenizer.model_max_length
#     train_tokens = tokenizer(sentiment_train_data['sentence'].to_list(), padding='max_length', max_length=max_length, truncation=True, return_tensors='pt')
#     test_tokens = tokenizer(sentiment_test_data['sentence'].to_list(), padding='max_length', max_length=max_length, truncation=True, return_tensors='pt')
#     train_set = LoadDatasetLLM(train_tokens, sentiment_train_data['sentiment'])
#     test_set = LoadDatasetLLM(test_tokens, sentiment_test_data['sentiment'])

#     train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
#     test_loader = DataLoader(test_set, batch_size=16)
    
    model= SentimentAnalysisModel('bert-base-uncased',3)
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    criterion= nn.CrossEntropyLoss()
    train_(model, optimizer, train_loader, criterion, device)
    
    


def train(args, config):
    seed_everything()
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Using device:", device)
    
    print("Starting: ", args.task, "learning")
    print("Using", args.transformer, "base pretrained model and tokenizer")
    
    train_dataloaders, test_dataloaders = load_dataloaders(args, config)
    
    print("All Data Loaded")
    
#     check(device, train_dataloaders['sentiment'])
#     current_dir = os.path.dirname(__file__)
#     current_dir = os.path.join(current_dir, '..', "Dataset")
#     sentiment_train_data=load_sentiment_data(os.path.join(current_dir,"Sentiment SST5/train.csv"))
#     sentiment_test_data=load_sentiment_data(os.path.join(current_dir,"Sentiment SST5/test.csv"))
#     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

#     max_length = tokenizer.model_max_length
#     train_tokens = tokenizer(sentiment_train_data['sentence'].to_list(), padding='max_length', max_length=max_length, truncation=True, return_tensors='pt')
#     test_tokens = tokenizer(sentiment_test_data['sentence'].to_list(), padding='max_length', max_length=max_length, truncation=True, return_tensors='pt')
#     train_set = LoadDatasetLLM(train_tokens, sentiment_train_data['sentiment'])
#     test_set = LoadDatasetLLM(test_tokens, sentiment_test_data['sentiment'])

#     train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
#     test_loader = DataLoader(test_set, batch_size=16)
    
#     if args.reproduce:
#         reproduce_results(test_dataloaders, device, args)
#         return   
    
#     train_dataloaders['sentiment']=train_loader
#     test_dataloaders['sentiment']=test_loader
    
    num_batches_per_epoch = 0
    
    scheduler = None
    
    
    if args.task== 'multitask':
        model= MultiTaskSentimentAnalysis(args, config)
        model.to(device)
        optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
        scaler = GradScaler() 
        objects_group = ObjectsGroup(model, optimizer, scaler)
        scheduler = RoundRobinScheduler(train_dataloaders)
        logs_file = os.path.join(args.logs_path, str(args.task+ "_" + args.transformer + '.txt'))
        
        with open(logs_file, "w") as f:
                f.write(f"Epoch,Training Loss Sentiment,Training Loss Sarcasm,Training Loss Negation,\
                Training Loss Similarity,Test Accuracy Sentiment,Test Accuracy Sarcasm,\
                Test Accuracy Negation,Test R2 Similarity\n")
        
    elif args.task=="individual": 
        for task in ['sentiment', 'neg', 'sar', 'sim']:
        # If we are using a single task, we don't need a scheduler
            scheduler = RandomScheduler(train_dataloaders)
            best_test_metric = -np.inf
            logs_file = os.path.join(args.logs_path, str(task+ "_" + args.transformer + '.txt'))
            
            with open(logs_file, "w") as f:
                if task=='sim':
                    f.write(f"Epoch,Training Loss,Test R2\n")
                else:
                    f.write(f"Epoch,Training Loss,Test Acc\n")
                    
 
            num_batches_per_epoch = int(len(train_dataloaders[task]) / config.gradient_accumulations[task])
    
            model= MultiTaskSentimentAnalysis(args, config)
            model.to(device)
            optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
            scaler = GradScaler() 
            objects_group = ObjectsGroup(model, optimizer, scaler)
            
#             for batch in train_dataloaders['sentiment']:
#                 optimizer.zero_grad()

#                 b_ids, b_mask, b_labels = (batch['token_ids'], batch['attention_mask'], batch['labels'])
#                 b_ids, b_mask, b_labels = b_ids.to(device), b_mask.to(device), b_labels.to(device)
#                 embeddings = model(b_ids, b_mask, task_id=0)
#                 logits = model.last_layers_sentiment(embeddings)

#                 criterion = nn.CrossEntropyLoss()
#                 loss = criterion(logits, b_labels)/config.batch_size
#                 print(loss)
#                 loss_value = loss.item()
#         #         scaler.scale(loss).backward()
#                 loss.backward()
#                 optimizer.step()
#                 optimizer.zero_grad(set_to_none=True)
# #                     torch.cuda.empty_cache()

            print("Training ", task , "data")
            for epoch in range(config.epochs):
                model.train()
                loss = 0
                for i in tqdm(range(num_batches_per_epoch)):
                    loss += scheduler.process_named_batch(objects_group, config, task, device)
                    
                loss/=len(train_dataloaders[task])
                # Evaluate on test set
                test_metric = model_eval_individual(test_dataloaders[task], task, model, device, args)

                with open(logs_file, "a") as f:
                    f.write(f"{epoch},{loss},{test_metric}\n")
                if task=='sim':
                    print(f"Epoch: {epoch},Training Loss: {loss},Test R2: {test_metric}")
                else:
                    print(f"Epoch: {epoch},Training Loss: {loss},Test Accuracy: {test_metric}")       

                if test_metric > best_test_metric:
                    best_test_metric = test_metric
                    best_model = copy.deepcopy(model)
                    model_file = os.path.join(args.model_path, str(task + "_" + args.transformer + '.pt'))
                    save_model(model, optimizer, args, config, model_file)

            model_eval_individual(test_dataloaders[task], task, best_model, device, args, final=True)

        return
    
    else:
        print("Task Not Valid.")
        print("Task should be either multitask or individual")
        return
    
    
    
    best_test_sentiment_acc= -np.inf
    
    for task in ['sentiment', 'neg', 'sar', 'sim']:
        num_batches_per_epoch += int(len(train_dataloaders[task]) / config.gradient_accumulations[task])
    
    print("Training...")
    
    for epoch in range(config.epochs):
        model.train()
        train_loss = {'sentiment': 0, 'sar': 0, 'neg': 0, 'sim': 0}

        for i in tqdm(range(num_batches_per_epoch)):
            
                task, loss = scheduler.process_one_batch(objects_group, config, device)
                train_loss[task] += loss.item()
        
                    
        (sentiment_accuracy, 
         negation_accuracy, 
         sarcasm_accuracy, 
         similarity_r2) = model_eval_multitask(test_dataloaders, model, device, args)
        
        with open(logs_file, "a") as f:
                f.write(f"{epoch},{train_loss['sentiment']}, {train_loss['sar']}, {train_loss['neg']},\
                {train_loss['sim']}, {sentiment_accuracy}, {sarcasm_accuracy},\
                {negation_accuracy}, {similarity_r2}\n")
                          
        print(f"Epoch: {epoch},Sentiment Training Loss: {train_loss['sentiment']},Test Accuracy: {sentiment_accuracy}")     
        
        if(sentiment_accuracy>best_test_sentiment_acc):
            best_test_sentiment_acc = sentiment_accuracy
            best_model = copy.deepcopy(model)
            model_file= os.path.join(args.model_path, str("multitask"+ "_" + args.transformer + '.pt'))
            save_model(model, optimizer, args, config, model_file)
                
    model_eval_multitask(test_dataloaders, best_model, device, args, final=True)