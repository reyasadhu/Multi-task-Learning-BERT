from codes.train import train
import argparse, yaml
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Fine tune a bert model')
parser.add_argument('task', type=str, choices=('multitask', 'individual'))
parser.add_argument('--transformer', type=str, choices=('bert, roberta'), default='bert')
parser.add_argument('--hyperparam_file', type=str, default='hyperparameters.yaml', help='The name of the hyperparameter file.')
parser.add_argument('--model_path', type=str, default='./models/', help='Filepath for saved models.')
parser.add_argument('--logs_path', type=str, default='./logs/', help='Filepath for logs')
parser.add_argument('--reproduce', type=bool, default=False, help='Make true to see results of trained model')
parser.add_argument("--use_gpu", action='store_true')


args = parser.parse_args()
with open(args.hyperparam_file) as f:
    hyperparams = yaml.load(f, Loader=yaml.FullLoader)

class ModelConfig:
    def __init__(self, hyperparams):
        self.batch_size = int(hyperparams.get('batch_size', 32))
        self.learning_rate = float(hyperparams.get('learning_rate', 2e-5))
        self.hidden_dropout_prob = float(hyperparams.get('dropout', 0.2))
        self.n_hidden_layers = int(hyperparams.get('hidden_layers', 1 ))
        self.epochs = int(hyperparams.get('epochs', 10))
        self.grad_clip = float(hyperparams.get('grad_clip', 1.0))
        self.max_batch_size_sentiment = int(hyperparams.get('max_batch_size_sentiment',32))
        self.max_batch_size_negation = int(hyperparams.get('max_batch_size_negation',16))
        self.max_batch_size_sarcasm = int(hyperparams.get('max_batch_size_sarcasm',32))
        self.max_batch_size_similarity = int(hyperparams.get('max_batch_size_similarity',16))
        self.num_batches_per_epoch = int(hyperparams.get('num_batches_per_epoch',0))
        self.n_sentiment_classes = 3
        self.gradient_accumulations={'sentiment' : 1, 'sar': 1, 'neg': 1, 'sim': 1}
        self.max_batch_size = {'sentiment' : self.max_batch_size_sentiment, 'sar': self.max_batch_size_sarcasm, 'neg': self.max_batch_size_negation, 'sim': self.max_batch_size_similarity}

        
        
        
config = ModelConfig(hyperparams)
train(args, config)
