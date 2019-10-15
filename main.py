import torch
import time

from config import Config
from data import load_dataset
from models import StyleTransformer, Discriminator
from train import train, auto_eval


def main():
    config = Config()
    train_iters, dev_iters, test_iters, vocab = load_dataset(config)
    print('Vocab size:', len(vocab))
    model_F = StyleTransformer(config, vocab).to(config.device)
    model_D = Discriminator(config, vocab).to(config.device)
    print(config.discriminator_method)
    
    train(config, vocab, model_F, model_D, train_iters, dev_iters, test_iters)
    

if __name__ == '__main__':
    main()
