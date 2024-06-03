import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel, RobertaTokenizer, RobertaModel
import warnings
warnings.filterwarnings("ignore")

class MultiTaskSentimentAnalysis(nn.Module):
    '''
    This module should use BERT for 4 tasks:

    - Sentiment classification (predict_sentiment)
    - Sarcasm detection (predict_sarcasm)
    - Negation Detection (predict_negation)
    - Semantic Textual Similarity (predict_similarity)
    '''
    def __init__(self, args, config):
        super(MultiTaskSentimentAnalysis, self).__init__()

        if args.transformer == 'roberta':
            self.bert = RobertaModel.from_pretrained("roberta-base")
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        if args.transformer == 'bert':
            self.bert = BertModel.from_pretrained("bert-base-uncased")
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            
        BERT_HIDDEN_SIZE =  self.bert.config.hidden_size
#         for param in self.bert.parameters():
#                 param.requires_grad = True
        
#         # Layers for sentiment classification
        self.dropout_sentiment = nn.ModuleList([nn.Dropout(config.hidden_dropout_prob) for _ in range(config.n_hidden_layers + 1)])
        self.linear_sentiment = nn.ModuleList(
    [nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size) for _ in range(config.n_hidden_layers)] + 
    [nn.Linear(self.bert.config.hidden_size, config.n_sentiment_classes)])
#         self.drop = nn.Dropout(p=0.3)
#         self.classifier = nn.Sequential(
#             nn.Linear(BERT_HIDDEN_SIZE, BERT_HIDDEN_SIZE),
#             nn.ReLU(),
#             nn.Dropout(p=0.3),
#             nn.Linear(BERT_HIDDEN_SIZE, config.n_sentiment_classes)
#         )
        # Layers for sarcasm detection
        self.dropout_sarcasm = nn.ModuleList([nn.Dropout(config.hidden_dropout_prob) for _ in range(config.n_hidden_layers + 1)])
        self.linear_sarcasm = nn.ModuleList([nn.Linear(BERT_HIDDEN_SIZE, BERT_HIDDEN_SIZE) for _ in range(config.n_hidden_layers)] + [nn.Linear(BERT_HIDDEN_SIZE, 1)])
        
        # Layers for negation detection
        self.dropout_negation = nn.ModuleList([nn.Dropout(config.hidden_dropout_prob) for _ in range(config.n_hidden_layers + 1)])
        self.linear_negation = nn.ModuleList([nn.Linear(BERT_HIDDEN_SIZE, BERT_HIDDEN_SIZE) for _ in range(config.n_hidden_layers)] + [nn.Linear(BERT_HIDDEN_SIZE, 1)])

        # Layers for sentence similarity
        self.dropout_similarity = nn.ModuleList([nn.Dropout(config.hidden_dropout_prob) for _ in range(config.n_hidden_layers + 1)])
        self.linear_similarity = nn.ModuleList([nn.Linear(BERT_HIDDEN_SIZE, BERT_HIDDEN_SIZE) for _ in range(config.n_hidden_layers)] + [nn.Linear(BERT_HIDDEN_SIZE, 1)])
        

    def forward(self, input_ids, attention_mask, task_id):
        'Takes a batch of sentences and produces embeddings for them.'
        bert_output = self.bert(input_ids, attention_mask)

        # Get the [CLS] token embeddings
        cls_embeddings = bert_output['pooler_output']

        return cls_embeddings


    def last_layers_sentiment(self, x):

        for i in range(len(self.linear_sentiment) - 1):
            x = self.dropout_sentiment[i](x)
            x = self.linear_sentiment[i](x)
            x = F.relu(x)
            
        x = self.dropout_sentiment[-1](x)
        logits = self.linear_sentiment[-1](x)
        return logits
    
    def predict_sentiment(self, input_ids, attention_mask):
        '''Given a batch of sentences, outputs logits for classifying sentiment.
        There are 3 sentiment classes:
        (0 - negative, 1- neutral, 2- positive)
        '''
        x = self.forward(input_ids, attention_mask, task_id=0)
        x = self.last_layers_sentiment(x)

        return x
    
    def last_layers_sarcasm(self, x):

        for i in range(len(self.linear_sarcasm) - 1):
            x = self.dropout_sarcasm[i](x)
            x = self.linear_sarcasm[i](x)
            x = F.relu(x)

        x = self.dropout_sarcasm[-1](x)
        logits = self.linear_sarcasm[-1](x)
        return logits

    def predict_sarcasm(self, input_ids, attention_mask):
        '''Given a batch of sentences, outputs logits for classifying sarcasm.
        (0 - not sarcastic, 1- sarcastic)
        '''
        x = self.forward(input_ids, attention_mask, task_id=1)
        x = self.last_layers_sarcasm(x)

        return x

    def get_paired_embeddings(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2, task_id):
        '''Given a batch of pairs of sentences, get the embeddings.'''
        # Get [SEP] token ids
        sep_token_id = torch.tensor([self.tokenizer.sep_token_id], dtype=torch.long, device=input_ids_1.device)
        batch_sep_token_id = sep_token_id.repeat(input_ids_1.shape[0], 1)

        # Step 1: Concatenate the two sentences in: sent1 [SEP] sent2 [SEP]
        input_id = torch.cat((input_ids_1, batch_sep_token_id, input_ids_2, batch_sep_token_id), dim=1)
        attention_mask = torch.cat((attention_mask_1, torch.ones_like(batch_sep_token_id), attention_mask_2, torch.ones_like(batch_sep_token_id)), dim=1)

        x = self.forward(input_id, attention_mask, task_id=task_id)

        return x
    
    def last_layers_negation(self, x):
        """Given a batch of pairs of sentences embedding, outputs logits for predicting whether they are negation or parapharases."""

        for i in range(len(self.linear_negation) - 1):
            x = self.dropout_negation[i](x)
            x = self.linear_negation[i](x)
            x = F.relu(x)

        x = self.dropout_negation[-1](x)
        logits = self.linear_negation[-1](x)
        return logits
    
    def predict_negation(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs whether its negation or not'''
        x = self.get_paired_embeddings(input_ids_1, attention_mask_1, input_ids_2, attention_mask_2, task_id=2)
        return self.last_layers_negation(x)
   


    def last_layers_similarity(self, x):
        """Given a batch of pairs of sentences embeddings, outputs logits for predicting how similar they are."""
        for i in range(len(self.linear_similarity) - 1):
            x = self.dropout_similarity[i](x)
            x = self.linear_similarity[i](x)
            x = F.relu(x)

      
        x = self.dropout_similarity[-1](x)
        preds = self.linear_similarity[-1](x)
        preds = torch.sigmoid(preds)  # Sigmoid to constrain to (0, 1)
        preds = preds * 5 
        return preds
    
    def predict_similarity(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit corresponding to how similar they are.'''
        # Get the embeddings
        x = self.get_paired_embeddings(input_ids_1, attention_mask_1, input_ids_2, attention_mask_2, task_id=3)
        return self.last_layers_similarity(x)