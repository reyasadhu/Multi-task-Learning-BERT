import os
import torch

def reproduce_results(dataloader, device, args):
    print("Without Multitasking")
    model_file= os.path.join(args.model_path, str("sentiment"+ "_" + args.transformer + '.pt'))
    checkpoint=torch.load(model_file, map_location=device)
    new_state_dict = {}
    for key, value in checkpoint['model'].items():
        new_key = key.replace("_orig_mod.", "")
        new_state_dict[new_key] = value
    model.load_state_dict(new_state_dict)
    optimizer.load_state_dict(checkpoint['optimizer'])
    args = checkpoint['args']
    model_eval_sentiment(dataloader, model, device, args, final=True)
    
    print("With Multitasking")
    model_file= os.path.join(args.model_path, str("multitask"+ "_" + args.transformer + '.pt'))
    checkpoint=torch.load(model_file, map_location=device)
    new_state_dict = {}
    for key, value in checkpoint['model'].items():
        new_key = key.replace("_orig_mod.", "")
        new_state_dict[new_key] = value
    model.load_state_dict(new_state_dict)
    optimizer.load_state_dict(checkpoint['optimizer'])
    args = checkpoint['args']
    model_eval_sentiment(dataloader, model, device, args, final=True)
    