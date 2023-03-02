import torch

def check_model_key_with_weight(model, weight):
    model_keys = [n for n, _ in model.named_parameters()]
    


def load(model, weights):
    print(weights['backbone'].keys())
    for n, p in model.named_parameters():
        print(n)
        
    exit()
    