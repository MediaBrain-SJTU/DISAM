import sys
sys.path.append(sys.path[0].replace('runs', ''))
import warnings
warnings.filterwarnings("ignore")
import argparse
import algorithms
algorithm_names = sorted(name for name in algorithms.__dict__ if 'Trainer' in name and callable(algorithms.__dict__[name]))

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", type=str, default='DG_CLIP_Zero_Shot_Trainer', choices=algorithm_names, help='Name of the algorithms')
    
    parser.add_argument("--backbone", type=str, default='resnet50', help='model name')
    
    parser.add_argument("--sub_log_dir", type=str, default='none')
    
    parser.add_argument('--seed', help='default seed', type=int, default=0)
    
    parser.add_argument('--weight_decay', help='weight_decay', type=float, default=0.0005)
    
    '''Dataset'''
    parser.add_argument("--dataset", type=str, default='pacs', help='Name of dataset')
    parser.add_argument('--dataset_seed', help='default seed for dataset split', type=int, default=0)

    parser.add_argument("--test_domain", type=str, default='p', help='the domain name for testing')
    
    '''Hyper-parameter'''
    parser.add_argument('--batch_size', help='batch_size', type=int, default=32)
    parser.add_argument('--val_batch_size', help='val_batch_size', type=int, default=32)
    parser.add_argument('--epochs', help='epochs number on MSDG', type=int, default=20)
    parser.add_argument('--optim', help='optimizer name', type=str, default='sgd')
    parser.add_argument('--lr', help='learning rate', type=float, default=0.002)
    parser.add_argument("--lr_policy", type=str, default='step', help="learning rate scheduler policy")

    # SAM
    parser.add_argument('--rho', help='rho', type=float, default=0.01)

    # GSAM SAGM
    parser.add_argument('--sam_alpha', help='sam_alpha', type=float, default=0.0)


    # DISAM
    parser.add_argument('--lambda_weight', help='lambda value for DISAM', type=float, default=0.1)
    
    '''logging'''
    parser.add_argument('--note', help='note of experimental settings', type=str, default='fedavg')
    parser.add_argument('--display', help='display in controller', action='store_true') # 默认false 即不展示
    return parser.parse_args()


def main():
    args = get_args()
    trainer = getattr(algorithms, args.algorithm)(args)
    trainer.run()

if __name__ == '__main__':
    main()