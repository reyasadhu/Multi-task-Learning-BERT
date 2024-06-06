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
    

def train(args, config):
    seed_everything()
    
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print("Using device:", device)
    
    print("Starting: ", args.task, "learning")
    print("Using", args.transformer, "base pretrained model and tokenizer")
    
    train_dataloaders, test_dataloaders = load_dataloaders(args, config)
    
    print("All Data Loaded")
    
    
    num_batches_per_epoch = 0
    
    scheduler = None
    
        
    if args.task=="individual": 
        for task in ['sentiment','neg','sar']:
        # If we are using a single task, we don't need a scheduler
            scheduler = RandomScheduler(train_dataloaders)
            best_test_metric = -np.inf
            logs_file = os.path.join(args.logs_path, str(task+ "_" + args.transformer + '.txt'))
            
            with open(logs_file, "w") as f:
                f.write(f"Epoch,Training Loss,Test Acc\n")
                    
 
            num_batches_per_epoch = int(len(train_dataloaders[task]) / config.gradient_accumulations[task])
    
            model= MultiTaskSentimentAnalysis(args, config)
            model.to(device)
            optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
            scaler = GradScaler() 
            objects_group = ObjectsGroup(model, optimizer, scaler)
        
            print("Training ", task , "data")
            for epoch in range(config.epochs):
                model.train()
                loss = 0
                for _ in tqdm(range(num_batches_per_epoch)):
                    loss += scheduler.process_named_batch(objects_group, config, task, device)
                    
                loss=loss/len(train_dataloaders[task])
                # Evaluate on test set
                test_metric = model_eval_individual(test_dataloaders[task], task, model, device, args)

                with open(logs_file, "a") as f:
                    f.write(f"{epoch+1},{loss},{test_metric}\n")


                print(f"Epoch: {epoch+1},Training Loss: {loss},Test Accuracy: {test_metric}%")       

                if test_metric > best_test_metric:
                    best_test_metric = test_metric
                    best_model = copy.deepcopy(model)
                    model_file = os.path.join(args.model_path, str(task + "_" + args.transformer + '.pt'))
                    save_model(model, optimizer, args, config, model_file)

            model_eval_individual(test_dataloaders[task], task, best_model, device, args, final=True)

        return
    
    elif args.task=="multitask":
        # Individual Pretraining
        model= MultiTaskSentimentAnalysis(args, config, pretraining=True)
        model.to(device)
        num_batches_per_epoch=len(train_dataloaders['sentiment'])
        for task in ['sentiment', 'neg', 'sar']:
            optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
            scheduler = RandomScheduler(train_dataloaders)
            scaler = GradScaler() 
            objects_group=ObjectsGroup(model, optimizer, scaler)
            
            for epoch in range(config.epochs):
                model.train()
                for _ in tqdm(range(num_batches_per_epoch)):
                    loss=scheduler.process_named_batch(objects_group, config, task, device)

        #Fine-tuning
        pretrained_model = copy.deepcopy(model.state_dict())
        model = MultiTaskSentimentAnalysis(args, config)
        model.load_state_dict(pretrained_model)
        model.to(device)
        optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
        scheduler = RoundRobinScheduler(train_dataloaders)
        objects_group=ObjectsGroup(model, optimizer, scaler)
            
        best_test_sentiment_acc= -np.inf
        
        for task in ['sentiment', 'neg', 'sar']:
            num_batches_per_epoch += int(len(train_dataloaders[task]) / config.gradient_accumulations[task])
        
        print("Training...")

        logs_file = os.path.join(args.logs_path, str(args.task+ "_" + args.transformer + '.txt'))
        
        with open(logs_file, "w") as f:
                f.write(f"Epoch,Training Loss Sentiment,Training Loss Sarcasm,Training Loss Negation,Test Accuracy Sentiment,Test Accuracy Sarcasm,Test Accuracy Negation\n")
        
        for epoch in range(config.epochs):
            model.train()
            train_loss = {'sentiment': 0, 'sar': 0, 'neg': 0}

            for _ in tqdm(range(num_batches_per_epoch)):
                    task, loss = scheduler.process_one_batch(objects_group, config, device)
                    train_loss[task] += loss
            
            for task in ['sentiment', 'neg', 'sar']:
                train_loss[task]/=len(train_dataloaders[task])

            (sentiment_accuracy, negation_accuracy, sarcasm_accuracy) = model_eval_multitask(test_dataloaders, model, device, args)
            
            with open(logs_file, "a") as f:
                    f.write(f"{epoch+1},{train_loss['sentiment']}, {train_loss['sar']}, {train_loss['neg']}, {sentiment_accuracy}, {sarcasm_accuracy},{negation_accuracy}\n")
                            
            print(f"Epoch: {epoch+1},Sentiment Training Loss: {train_loss['sentiment']},Test Accuracy: {sentiment_accuracy}")     
            
            if(sentiment_accuracy>best_test_sentiment_acc):
                best_test_sentiment_acc = sentiment_accuracy
                best_model = copy.deepcopy(model)
                model_file= os.path.join(args.model_path, str("multitask"+ "_" + args.transformer + '.pt'))
                save_model(model, optimizer, args, config, model_file)
                    
        model_eval_multitask(test_dataloaders, best_model, device, args, final=True)
    
    else:
        print("Task Not Valid.")
        print("Task should be either multitask or individual")
        return