import torch
import yaml
import os
import pandas as pd
import argparse


device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')


def save_checkpoint(model, model_name):
    ckpt = {'model_weights': model.state_dict()}
    torch.save(ckpt, f"train_checkpoint/{model_name}_ckpt.pth")


def load_checkpoint(model, file_name):
    ckpt = torch.load(file_name, map_location=device)
    model_weights = ckpt['model_weights']
    model.load_state_dict(model_weights)
    print("Model's pretrained weights loaded!")


def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config