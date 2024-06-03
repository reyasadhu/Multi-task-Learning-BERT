import torch
import os
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, r2_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
from plot_curves import plot
import numpy as np
import pandas as pd

def model_eval_sentiment(dataloader, model, device, args, final):

    model.eval()  
    y_true = []
    y_pred = []
    sents = []

    for batch in dataloader:
        b_ids, b_mask, b_labels, b_sents= batch['token_ids'],batch['attention_mask'], batch['labels'], batch['sents']

        b_ids = b_ids.to(device)
        b_mask = b_mask.to(device)

        logits = model.predict_sentiment(b_ids, b_mask)
        logits = logits.detach().cpu().numpy()
        preds = np.argmax(logits, axis=1).flatten()

        b_labels = b_labels.flatten()
        y_true.extend(b_labels)
        y_pred.extend(preds)
        sents.extend(b_sents)

    sentiment_acc = accuracy_score(y_true, y_pred)
    
    if final==True:
        print(f"Final Sentiment Accuracy {args.task} learning:",sentiment_acc)
        cm = confusion_matrix(y_true, y_pred)
        class_names = ['Negative', 'Neural', 'Positive']
        cm = confusion_matrix(y_true, y_pred, labels=[0,1,2])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot() 
        plt.show()
            
        class_report = classification_report(y_true, y_pred, target_names=class_names)
        print(f"\nClassification Report Sentiment {args.task} learning - {args.transformer}:")
        print(class_report)
        
        if not args.reproduce:
            current_dir = os.path.dirname(__file__)
            results_dir = os.path.join(current_dir, '..', 'results')
            plt.savefig(os.path.join(results_dir,f'Confusion Matrix Sentiment {args.task} learning {args.transformer}.png'))
            data = {'sentence': sents, 'True_label': y_true, 'Pred_label': y_pred}
            df = pd.DataFrame(data)
            df.to_csv(f"./results/Sentiment_prediction_{args.task}_learning_{args.transformer}.csv")
            if args.task!='multitask': 
                    plot(args)

    return sentiment_acc



def model_eval_negation(neg_dataloader, model, device, args, final):

    model.eval()  

    with torch.no_grad():
        neg_y_true = []
        neg_y_pred = []


        for batch in neg_dataloader:
            b_ids1, b_mask1, b_ids2, b_mask2, b_labels = (batch['token_ids_1'], batch['attention_mask_1'],
                          batch['token_ids_2'], batch['attention_mask_2'],
                          batch['labels'])

            b_ids1 = b_ids1.to(device)
            b_mask1 = b_mask1.to(device)
            b_ids2 = b_ids2.to(device)
            b_mask2 = b_mask2.to(device)

            logits = model.predict_negation(b_ids1, b_mask1, b_ids2, b_mask2)
            y_hat = logits.sigmoid().round().flatten().cpu().numpy()
            b_labels = b_labels.flatten().cpu().numpy()

            neg_y_pred.extend(y_hat)
            neg_y_true.extend(b_labels)

        neg_accuracy = np.mean(np.array(neg_y_pred) == np.array(neg_y_true))
        
        if final==True:
            print(f"Final Negation Accuracy {args.task} learning:",neg_accuracy)
            cm = confusion_matrix(neg_y_true, neg_y_pred)
            class_names = ['Paraphrase', 'Negation']
            cm = confusion_matrix(y_true, y_pred, labels=[0,1])
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
            disp.plot() 
            plt.show()

            if not args.reproduce:
                current_dir = os.path.dirname(__file__)
                results_dir = os.path.join(current_dir, '..', 'results')
                plt.savefig(os.path.join(results_dir,f"Confusion Matrix {args.task} learning {args.transformer}.png"))
                if args.task!='multitask': 
                    plot(args)

            class_report = classification_report(neg_y_true, neg_y_pred, target_names=class_names)
            print(f"\nClassification Report Negation {args.task} learning - {args.transformer}:")
            print(class_report)
            
            
        return neg_accuracy



def model_eval_similarity(sim_dataloader, model, device, args, final):
    model.eval()  

    with torch.no_grad():
        sim_y_true = []
        sim_y_pred = []

        for batch in sim_dataloader:
            b_ids1, b_mask1, b_ids2, b_mask2,b_labels = (batch['token_ids_1'], batch['attention_mask_1'],
                          batch['token_ids_2'], batch['attention_mask_2'],
                          batch['labels'])

            b_ids1 = b_ids1.to(device)
            b_mask1 = b_mask1.to(device)
            b_ids2 = b_ids2.to(device)
            b_mask2 = b_mask2.to(device)

            logits = model.predict_similarity(b_ids1, b_mask1, b_ids2, b_mask2)
            y_hat = logits.flatten().cpu().numpy()
            b_labels = b_labels.flatten().cpu().numpy()

            sim_y_pred.extend(y_hat)
            sim_y_true.extend(b_labels)
 
        r2_value = r2_score(sim_y_true, sim_y_pred).item()
    
        if final==True:
            print(f"R2 score similarity {args.task} learning {args.transformer}:", r2_value)
            
        if not args.reproduce:
            if args.task!='multitask': 
                    plot(args)
            
        return r2_value



def model_eval_sarcasm(sar_dataloader, model, device, args, final):
    model.eval()  

    with torch.no_grad():
        sar_y_true = []
        sar_y_pred = []


        for step, batch in sar_dataloader:
            b_ids, b_mask, b_labels = batch['token_ids'], batch['attention_mask'], batch['labels']

            b_ids = b_ids.to(device)
            b_mask = b_mask.to(device)

            logits = model.predict_sentiment(b_ids, b_mask)
            y_hat = logits.argmax(dim=-1).flatten().cpu().numpy()
            b_labels = b_labels.flatten().cpu().numpy()

            sar_y_pred.extend(y_hat)
            sar_y_true.extend(b_labels)


        sarcasm_accuracy = np.mean(np.array(sar_y_pred) == np.array(sar_y_true))
        
        if final==True:
            print(f"Final Sarcasm Accuracy {args.task} learning:",sarcasm_accuracy)
            cm = confusion_matrix(sar_y_true, sar_y_pred)
            class_names = ['Not Sarcastic', 'Sarcastic']
            cm = confusion_matrix(y_true, y_pred, labels=[0,1])
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
            disp.plot() 
            plt.show()

            if not args.reproduce:
                current_dir = os.path.dirname(__file__)
                results_dir = os.path.join(current_dir, '..', 'results')
                plt.savefig(os.path.join(results_dir,f"Confusion Matrix Sarcasm {args.task} learning {args.transformer}.png"))
                if args.task!='multitask': 
                    plot(args)

            class_report = classification_report(sar_y_true, sar_y_pred, target_names=class_names)
            print(f"\nClassification Report Sarcasm {args.task} learning - {args.transformer}:")
            print(class_report)
            
        return sarcasm_accuracy
    
def model_eval_individual(dataloader, task, model, device, args, final=False):    
    if task == 'sentiment': test_metric = model_eval_sentiment(dataloader, model, device, args, final)
    elif task == 'neg': test_metric = model_eval_negation(dataloader, model, device, args, final)
    elif task == 'sar': test_metric = model_eval_sarcasm(dataloader, model, device, args, final)
    elif task=='sim': test_metric = model_eval_similarity(dataloader, model, device, args, final)
    return test_metric


def model_eval_multitask(dataloaders, model, device, args, final=False):
    
    if final==True:
        plot(args)
        
    sentiment_accuracy = model_eval_sentiment(dataloaders['sentiment'], model, device, args, final)
    negation_accuracy = model_eval_negation(dataloaders['negation'], model, device, args, final)
    sarcasm_accuracy = model_eval_sarcasm(dataloaders['sarcasm'], model, device, args, final)
    similarity_r2 = model_eval_similarity(dataloaders['similarity'], model, device, args, final)
    

    return (sentiment_accuracy, negation_accuracy, sarcasm_accuracy, similarity_r2)
