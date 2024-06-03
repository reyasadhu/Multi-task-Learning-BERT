import torch
from individual_training import process_sentiment_batch, process_negation_batch, process_sarcasm_batch, process_similarity_batch
from itertools import cycle
torch.autograd.set_detect_anomaly(True)
import warnings
warnings.filterwarnings("ignore")

class ObjectsGroup:

    def __init__(self, model, optimizer, scaler):
        self.model = model
        self.optimizer = optimizer
        self.scaler = scaler
        self.loss_sum = 0

class Scheduler:

    def __init__(self, dataloaders, reset=True):
        self.dataloaders = dataloaders
        self.names = list(dataloaders.keys())
        if reset: self.reset()

    def reset(self):
        self.sentiment_iter = iter(self.dataloaders['sentiment'])
        self.sar_iter = iter(self.dataloaders['sar'])
        self.neg_iter = iter(self.dataloaders['neg'])
        self.sim_iter = iter(self.dataloaders['sim'])
        self.steps = {'sentiment': 0, 'sar': 0, 'neg': 0, 'sim': 0}

    def get_sentiment_batch(self):
        try:
            return next(self.sentiment_iter)
        except StopIteration:
            self.sentiment_iter = cycle(self.dataloaders['sentiment'])
            return next(self.sentiment_iter)

    def get_sarcasm_batch(self):
        try:
            return next(self.sar_iter)
        except StopIteration:
            self.sar_iter = cycle(self.dataloaders['sar'])
            return next(self.para_iter)

    def get_negation_batch(self):
        try:
            return next(self.neg_iter)
        except StopIteration:
            self.neg_iter = cycle(self.dataloaders['neg'])
            return next(self.neg_iter)
        
    def get_similarity_batch(self):
        try:
            return next(self.sim_iter)
        except StopIteration:
            self.sim_iter = cycle(self.dataloaders['sim'])
            return next(self.sim_iter)

    def get_batch(self, name: str):
        if name == "sentiment": return self.get_sentiment_batch()
        elif name == "sar": return self.get_sarcasm_batch()
        elif name == "neg": return self.get_negation_batch()
        elif name == "sim": return self.get_similarity_batch()
        raise ValueError(f"Unknown batch name: {name}")

    def process_named_batch(self, objects_group, config, name, device):
 
        batch = self.get_batch(name)
        process_fn, gradient_accumulation = None, config.gradient_accumulations[name]
        if name == "sentiment":
            process_fn = process_sentiment_batch
        elif name == "neg":
            process_fn = process_negation_batch
        elif name == "sar":
            process_fn = process_sarcasm_batch
        elif name == "sim":
            process_fn = process_similarity_batch
        else:
            raise ValueError(f"Unknown batch name: {name}")
        
        loss_of_batch = 0
        for _ in range(gradient_accumulation):
            loss_of_batch += process_fn(batch, objects_group, config, device)

        self.steps[name] += 1
        model, optimizer, scaler = objects_group.model, objects_group.optimizer, objects_group.scaler
        
        if config.grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        
        scaler.step(optimizer)
        scaler.update()
        
#         objects_group.optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        loss_value = objects_group.loss_sum
        objects_group.loss_sum = 0
        torch.cuda.empty_cache()

        return loss_of_batch
    
class RoundRobinScheduler(Scheduler):
    def __init__(self, dataloaders):
        super().__init__(dataloaders, reset=False)
        self.reset()

    def reset(self):
        self.index = 0
        return super().reset()

    def process_one_batch(self, objects_group, config):
        name = self.names[self.index]
        self.index = (self.index + 1) % len(self.names)
        return name, self.process_named_batch(objects_group, config, name)
    
class RandomScheduler(Scheduler):   
    '''A scheduler that randomly chooses a batch to process.'''
    def __init__(self, dataloaders):
        super().__init__(dataloaders, reset=True)

    def process_one_batch(self, objects_group, config, device):
        name = random.choice(self.names)
        return name, self.process_named_batch(objects_group, config, name, device)