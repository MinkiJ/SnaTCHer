import numpy as np
import torch
from model.trainer.fsl_trainer_SnaTCHerF import FSLTrainer
from model.utils import (
    set_gpu,
)

import argparse

import os
import sys
import random
sys.path.append(os.getcwd())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--episodes_per_epoch', type=int, default=600)
    parser.add_argument('--num_eval_episodes', type=int, default=600)
    parser.add_argument('--model_class', type=str, default='SnaTCHerF', 
                        choices=['SnaTCHerF', 'SnaTCHerT', 'SnaTCHerL'])
    parser.add_argument('--use_euclidean', type=bool, default=True)
    parser.add_argument('--backbone_class', type=str, default='Res12',
                        choices=['ConvNet', 'Res12', 'Res18'])
    parser.add_argument('--dataset', type=str, default='TieredImageNet',
                        choices=['MiniImageNet', 'TieredImageNet'])
    
    parser.add_argument('--closed_way', type=int, default=5)
    parser.add_argument('--closed_eval_way', type=int, default=5)
    
    parser.add_argument('--open_way', type=int, default=5)
    parser.add_argument('--open_eval_way', type=int, default=5)
    
    parser.add_argument('--shot', type=int, default=5)
    
    parser.add_argument('--query', type=int, default=15)
    parser.add_argument('--eval_query', type=int, default=15)
    parser.add_argument('--balance', type=float, default=0.01)
    parser.add_argument('--temperature', type=float, default=64)
    parser.add_argument('--temperature2', type=float, default=64)  # the temperature in the  
     
    # optimization parameters
    parser.add_argument('--orig_imsize', type=int, default=-1) # -1 for no cache, and -2 for no resize, only for MiniImageNet and CUB
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--lr_mul', type=float, default=10)    
    parser.add_argument('--lr_scheduler', type=str, default='step', choices=['multistep', 'step', 'cosine'])
    parser.add_argument('--step_size', type=str, default='20')
    parser.add_argument('--gamma', type=float, default=0.5)    
    parser.add_argument('--fix_BN', action='store_true', default=False)     # means we do not update the running mean/var in BN, not to freeze BN
    parser.add_argument('--augment',   action='store_true', default=False)
    parser.add_argument('--multi_gpu', action='store_true', default=False)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--init_weights', type=str, default=None)
    
    # usually untouched parameters
    parser.add_argument('--mom', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.0005) # we find this weight decay value works the best
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--eval_interval', type=int, default=1)
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--data_path', type=str, default=None)
    
    args = parser.parse_args()
    
    
    if args.dataset == 'MiniImageNet':
      args.data_path = './data/miniimagenet'
    else:
      args.data_path = './data/tiered-imagenet'
      
    if args.data_path is None:
      raise ValueError('Specify your data path')
      
    
    args.way = args.closed_way + args.open_way
    args.eval_way = args.way
    args.eval_shot = args.shot
    args.num_classes = args.way
    
    
    args.save_path = './checkpoints'
    prefix = 'mini' if args.dataset == 'MiniImageNet' else 'tiered'
    args.weight_name = '%s-feat-%d-shot.pth' % (prefix, args.shot)
    
    
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    set_gpu(args.gpu)
    
    trainer = FSLTrainer(args)
    trainer.evaluate_test()
