import os
import torch
import argparse
from torch.utils.data import DataLoader
from torch import nn

from asteroid import Container, Solver
from asteroid.filterbanks.stft_fb import STFTFB
from asteroid.masknn import TDConvNet
from asteroid.engine.losses import PITLossContainer, pairwise_neg_sisdr
from asteroid.engine.optimizers import make_optimizer

from dataset import WSJ2mixDataset

parser = argparse.ArgumentParser()
parser.add_argument('--use_cuda', type=int, default=0,
                    help='Whether use GPU')
parser.add_argument('--model_path', default='exp/tmp/final.pth',
                    help='Full path to save best validation model')

def main(conf):
    train_set = WSJ2mixDataset(conf['data']['wav_id'], \
            os.path.join(conf['data']['wav_base_path'], 'tr'),\
            sample_rate=conf['data']['sample_rate'],\
            segment=conf['data']['segment'])
    valid_set = WSJ2mixDataset(conf['data']['wav_id'], \
            os.path.join(conf['data']['wav_base_path'], 'cv'),\
            sample_rate=conf['data']['sample_rate'],\
            segment=conf['data']['segment'])

    train_loader = DataLoader(train_set, shuffle=True,
                              batch_size=conf['data']['batch_size'],
                              num_workers=conf['data']['num_workers'])
    val_loader = DataLoader(valid_set, shuffle=True,
                            batch_size=conf['data']['batch_size'],
                            num_workers=conf['data']['num_workers'])
    loaders = {'train_loader': train_loader, 'val_loader': val_loader}

    encoder = STFTFB(enc_or_dec='encoder', **conf['filterbank'])
    decoder = STFTFB(enc_or_dec='decoder', **conf['filterbank'])
    masker = TDConvNet(in_chan=encoder.n_feats_out,
                       out_chan=encoder.n_feats_out,
                       n_src=train_set.n_src, **conf['masknet'])
