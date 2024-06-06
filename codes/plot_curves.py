import os
import matplotlib.pyplot as plt
import pandas as pd

def plot(args):

    current_dir = os.path.dirname(__file__)
    results_dir = os.path.join(current_dir, '..', 'results')
        
    if args.task =='individual':
        for task in ['sentiment', 'neg', 'sar']:
            filepath= os.path.join(args.logs_path, str(task + "_" + args.transformer + '.txt'))
            try:
                df=pd.read_csv(filepath)
            except FileNotFoundError:
                print(f"File {filepath} not found!")
                continue

            plt.figure(figsize=(5, 10))

            plt.subplot(2, 1, 1)
            plt.plot(df['Epoch'], df['Training Loss'], marker='o')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(f'Training Loss')
    
            plt.subplot(2, 1, 2)
            plt.plot(df['Epoch'], df['Test Acc'], marker='o')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.title('Test Accuracy')
                

            plt.suptitle(f'Individual Training Results - {task}')

            plt.savefig(os.path.join(results_dir, f'{task}_individual_learning_plots.png'))

            
    else:
        filepath= os.path.join(args.logs_path, str("multitask" + "_" + args.transformer + '.txt'))
        try:
            df=pd.read_csv(filepath)
        except FileNotFoundError:
            print(f"File {filepath} not found!")
            return
       
        plt.figure(figsize=(15, 20))

        plt.subplot(3, 2, 1)
        plt.plot(df['Epoch'], df['Training Loss Sentiment'], marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Training Loss - Sentiment')
        
        
        plt.subplot(3, 2, 2)
        plt.plot(df['Epoch'], df['Test Accuracy Sentiment'], marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title(f'Test Accuracy - Sentiment')
        
            
        plt.subplot(3, 2, 3)
        plt.plot(df['Epoch'], df['Training Loss Sarcasm'], marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Training Loss - Sarcasm')
        
        
        plt.subplot(3, 2, 4)
        plt.plot(df['Epoch'], df['Test Accuracy Sarcasm'], marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title(f'Test Accuracy - Sarcasm')
        
    
        plt.subplot(3, 2, 5)
        plt.plot(df['Epoch'], df['Training Loss Negation'], marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Training Loss - Negation')
        
        
        plt.subplot(3, 2, 6)
        plt.plot(df['Epoch'], df['Test Accuracy Negation'], marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title(f'Test Accuracy - Negation')
        
        
        # plt.subplot(4, 2, 7)
        # plt.plot(df['Epoch'], df['Training Loss Similarity'], marker='o')
        # plt.xlabel('Epoch')
        # plt.ylabel('Loss')
        # plt.title(f'Training Loss - Similarity')
        
        
        # plt.subplot(4, 2, 8)
        # plt.plot(df['Epoch'], df['Test R2 Similarity'], marker='o')
        # plt.xlabel('Epoch')
        # plt.ylabel('R2 score')
        # plt.title('Test R2 score - Similarity')
                    
        plt.suptitle('Multitask Training Results')
        plt.show()
        
        plt.savefig(os.path.join(results_dir,'Multitask Plots.png'))