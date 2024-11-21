import argparse
import torch
from exp.exp_AD_2 import Exp_Anomaly_Detection
from exp.exp_Classifier import Exp_Anomaly_Classification
import random
import numpy as np

fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description='')

# model choose
parser.add_argument('--model', type=str,  default='recon',help='options: [classifier,recon,DPQAM64_recon,DPQAM64_classifier,LSTM_recon,LSTM_classifier]')
parser.add_argument('--modulation', type=int,  default=12,help='dp-qam64 ,a signal represent 12 bit')

# path config
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
parser.add_argument('--path', type=str, default='D:\lumin\pythonProject\dataset')

# dataset config
parser.add_argument('--iteration', type=int, default=270, help='optisystem iteration num')
parser.add_argument('--length', type=int, default=16384 , help='read length from file')

# size config
parser.add_argument('--seq_len', type=int, default=16384, help='first model input sequence length')
parser.add_argument('--seq_ch', type=int, default=2, help='first model input sequence channel size')
parser.add_argument('--reduce_len', type=int, default=16384 , help='after Reduce layer ,the channel size')
parser.add_argument('--reduce_ch', type=int, default=12 , help='after Reduce layer ,the channel size')
parser.add_argument('--out_len', type=int, default=16384, help='expected final output length')
parser.add_argument('--out_ch', type=int, default=2, help='expected final output size')


# model define
parser.add_argument('--dropout', type=float, default=0.3, help='dropout')
parser.add_argument('--d_model', type=int, default=128, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=2, help='num of heads')
parser.add_argument('--n_layers', type=int, default=2, help='num of layers')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--CNN_dim', type=int, default=16, help='num of CNN hidden_dim')
parser.add_argument('--d_ff', type=int, default=32, help='dimension of fcn [RWKV]')

# optimization
parser.add_argument('--train_epochs', type=int, default=1, help='train epochs')
parser.add_argument('--batch_size', type=int, default=16, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=8e-5, help='optimizer learning rate')
parser.add_argument('--loss', type=str, default='MSE', help='loss function')
parser.add_argument('--lradj', type=str, default='type9', help='adjust learning rate')
parser.add_argument('--delta', type=float, default='5e-4', help='adjust learning rate')

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--ratio', type=float, default=1.5)

if __name__ == '__main__':
    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    print('Args in experiment : ',args)

    Exp = Exp_Anomaly_Detection
    setting = '{}_{}_{}_{}_{}_{}'.format(
        args.model,
        args.seq_len,
        args.seq_ch,
        args.d_model,
        args.use_gpu,
        args.dropout
    )
    exp = Exp(args)
    print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
    exp.train(setting)
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting)
    torch.cuda.empty_cache()

    # exp_c = Exp_Anomaly_Classification(args)
    # print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
    # exp_c.train(setting)
    # print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    # exp_c.test(setting)
    # torch.cuda.empty_cache()

