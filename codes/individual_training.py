import torch
from torch import nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from contextlib import nullcontext
import warnings
warnings.filterwarnings("ignore")
torch.autograd.set_detect_anomaly(True)


def process_sentiment_batch(batch, objects_group, config, device):

    model, scaler = objects_group.model, objects_group.scaler

    with autocast() :
        b_ids, b_mask, b_labels = (batch['token_ids'], batch['attention_mask'], batch['labels'])
        b_ids, b_mask, b_labels = b_ids.to(device), b_mask.to(device), b_labels.to(device)
        embeddings = model(b_ids, b_mask, task_id=0)
        logits = model.last_layers_sentiment(embeddings)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, b_labels)/config.batch_size_individual['sentiment']
        loss_value = loss.item()*config.batch_size_individual['sentiment']
        objects_group.loss_sum += loss_value
        scaler.scale(loss).backward()
        return loss_value


def process_negation_batch(batch, objects_group, config, device):
    
    model, scaler = objects_group.model, objects_group.scaler

    with autocast() :
        b_ids_1, b_mask_1, b_ids_2, b_mask_2, b_labels = (batch['token_ids_1'], batch['attention_mask_1'], batch['token_ids_2'], batch['attention_mask_2'], batch['labels'])
        b_ids_1, b_mask_1, b_ids_2, b_mask_2, b_labels = b_ids_1.to(device), b_mask_1.to(device), b_ids_2.to(device), b_mask_2.to(device), b_labels.to(device)
        embeddings = model.get_paired_embeddings(b_ids_1, b_mask_1, b_ids_2, b_mask_2, task_id=2)
        preds = model.last_layers_negation(embeddings)
        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(preds.view(-1), b_labels.float()) / config.batch_size_individual['neg']
        loss_value = loss.item()*config.batch_size_individual['neg']

        objects_group.loss_sum += loss_value
        
        scaler.scale(loss).backward()

        return loss_value

def process_sarcasm_batch(batch, objects_group, config, device):
    model, scaler = objects_group.model, objects_group.scaler

    with autocast():
        b_ids, b_mask, b_labels = (batch['token_ids'], batch['attention_mask'], batch['labels'])
        b_ids, b_mask, b_labels = b_ids.to(device), b_mask.to(device), b_labels.to(device)

        embeddings = model(b_ids, b_mask, task_id=1)
        logits = model.last_layers_sarcasm(embeddings)
        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(logits.view(-1), b_labels.float()) / config.batch_size_individual['sar']
        loss_value = loss.item()*config.batch_size_individual['sar']

        objects_group.loss_sum += loss_value

        scaler.scale(loss).backward()

        return loss_value

