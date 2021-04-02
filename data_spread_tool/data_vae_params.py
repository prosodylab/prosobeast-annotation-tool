#!/usr/bin/env python3
"""
VAE model parameters.

@author:
    Branislav Gerazov Apr 2020
"""
import torch


class Params():
    """Deep VAE model parameters."""

    def __init__(self):

        # model parameters
        self.n_feats = None  # n of samples of f0 per noi
        # this will be overloaded with the length of the data

        # model type
        # self.model_type = 'dnn'  # available dnn or rnn
        # # from Report 04, best 2D performance is [64] * 4,
        # # but 3 gives better spread,
        # # for 4D it's 2 x 128, but 64 x 3 is also ok so we can take it
        # # as common
        # self.hidden_units = [64] * 3  # for the encoder and decoder
        # self.activation = 'tanh'  # tanh for VAE

        self.model_type = 'dnn'  # available dnn, rnn
        # self.model_type = 'rnn'  # available dnn, rnn
        self.n_nois = 2  # n of ramps to encode noi position
        # self.rnn_model = 'rnn'  # available rnn, gru, lstm
        self.rnn_model = 'lstm'  # available rnn, gru, lstm
        # self.rnn_model = 'gru'  # available rnn, gru, lstm
        # self.hidden_units = [32] * 2  # for the encoder and decoder
        self.hidden_units = [32]  # for the encoder and decoder

        self.n_latent = 4
        self.reg_type = 'mmd'  # can be kld or mmd
        self.reg_vae = 0
        # self.reg_vae = 0.001  # regularisation coeff of vae loss (kld or mmd)

        # training params
        self.verbose = False
        if torch.cuda.is_available():
            self.use_cuda = True
            print('GPU detected')
        else:
            print('No GPU detected')
            self.use_cuda = False
        self.max_iter = int(1e6)  # this is epochs for deep model
        # self.max_iter = int(1e1)  # this is epochs for deep model
        self.learn_rate = 0.001
        self.l2 = 0.0001  # 1e-4 default
        self.drop_rate = 0  # not used in the VAE
        self.optimizer_type = 'adam'
        # can be 'adam', 'sgd','rmsprop','rprop'

        self.batch_size = 512  # auto is min(n_samples, 64)
        # available int, auto and all
        if self.model_type == 'rnn':
            self.shuffle = False
        else:
            self.shuffle = True
        self.seed = 42
        self.early_stopping = True
        self.early_thresh = 1e-5  # change in RMS to decrease patience 1e-4
        # number of epochs to wait if there is no improvement
        self.patience = 500
        self.use_validation = False
        if not self.early_stopping:
            # split train data into train and val for early stopping
            self.use_validation = True
        # percentage of data to keep for validation
        self.validation_size = 0.1
