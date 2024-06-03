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

    with nullcontext():
        b_ids, b_mask, b_labels = (batch['token_ids'], batch['attention_mask'], batch['labels'])
        b_ids, b_mask, b_labels = b_ids.to(device), b_mask.to(device), b_labels.to(device)
        embeddings = model(b_ids, b_mask, task_id=0)
        logits = model.last_layers_sentiment(embeddings)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, b_labels)/config.batch_size
        print(loss)
        loss_value = loss.item()

        objects_group.loss_sum += loss_value

        scaler.scale(loss).backward()

        return loss


def process_negation_batch(batch, objects_group, config, device):
    print("negation")
    model, scaler = objects_group.model, objects_group.scaler

    with autocast():
        b_ids_1, b_mask_1, b_ids_2, b_mask_2, b_labels = (batch['token_ids_1'], batch['attention_mask_1'], batch['token_ids_2'], batch['attention_mask_2'], batch['labels'])
        b_ids_1, b_mask_1, b_ids_2, b_mask_2, b_labels = b_ids_1.to(device), b_mask_1.to(device), b_ids_2.to(device), b_mask_2.to(device), b_labels.to(device)

        embeddings = model.get_similarity_negation_embeddings(b_ids_1, b_mask_1, b_ids_2, b_mask_2, task_id=2)
        preds = model.last_layers_negation(embeddings)
        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(preds, b_labels) / config.batch_size
        loss_value = loss.item()

        objects_group.loss_sum += loss_value
        
        scaler.scale(loss).backward()

        return loss_value

def process_sarcasm_batch(batch, objects_group, config, device):
    print("sarcasm")
    model, scaler = objects_group.model, objects_group.scaler

    with autocast():
        b_ids, b_mask, b_labels = (batch['token_ids'], batch['attention_mask'], batch['labels'])
        b_ids, b_mask, b_labels = b_ids.to(device), b_mask.to(device), b_labels.to(device)

        embeddings = model(b_ids, b_mask, task_id=1)
        logits = model.last_layers_sarcasm(embeddings)
        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(logits, b_labels) / config.batch_size
        loss_value = loss.item()

        objects_group.loss_sum += loss_value

        scaler.scale(loss).backward()

        return loss_value

def process_similarity_batch(batch, objects_group, config, device):
    print("similarity")
    device = config.device
    model, scaler = objects_group.model, objects_group.scaler

    with autocast():
        b_ids_1, b_mask_1, b_ids_2, b_mask_2, b_labels = (batch['token_ids_1'], batch['attention_mask_1'], batch['token_ids_2'], batch['attention_mask_2'], batch['labels'])
        b_ids_1, b_mask_1, b_ids_2, b_mask_2, b_labels = b_ids_1.to(device), b_mask_1.to(device), b_ids_2.to(device), b_mask_2.to(device), b_labels.to(device)

        embeddings = model.get_similarity_negation_embeddings(b_ids_1, b_mask_1, b_ids_2, b_mask_2, task_id=3)
        preds = model.last_layers_similarity(embeddings)
        criterion = nn.MSELoss()
        loss = criterion(preds, b_labels) / config.batch_size
        loss_value = loss.item()

        objects_group.loss_sum += loss_value
        
        scaler.scale(loss).backward()
        return loss_value
