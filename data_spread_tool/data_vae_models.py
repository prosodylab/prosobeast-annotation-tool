#!/usr/bin/env python3
"""
VAE models for Data spread calculation.

@author:
    Branislav Gerazov Apr 2020
"""
import numpy as np
try:
    import torch
    import torch.nn.utils.rnn as rnn_utils
    from torch.utils.data.sampler import (
        SequentialSampler, RandomSampler, BatchSampler
        )
except ImportError:
    print('No PyTorch installed!')

# try:
#     # for running from parent dir
#     from data_spread_tool import data_vae_utils as vae_utils
# except ImportError:
#     # for running from current
#     import data_vae_utils as vae_utils


class RVAEModel(torch.nn.Module):
    """
    This is a recurrent variational encoder based on the Prosodeep VRNN.

    vae_as_input uses the latent space sample to generate a SOS input that
    initialises the hidden state for processing the real input. Like in:
    https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/03-advanced/image_captioning

    RNN cells:
    Inputs: input, hidden
        - **input** (batch, input_size): tensor containing input features
        - **hidden** (batch, hidden_size): tensor containing the initial hidden
          state for each element in the batch.

    Outputs: h'
        - **h'** (batch, hidden_size): tensor containing the next hidden state
          for each element in the batch
    """

    def __init__(
            self,
            n_feats,  # f0s per time step
            n_nois,  # ramps to encode the nois at time step (default 2)
            hidden_units,  # rnn hidden layer dims
            n_latent,  # latent space dims
            rnn_model='rnn',
            vae_as_input=False,  # use latent space as initial input
            ):
        super().__init__()

        self.n_feats = n_feats
        self.n_nois = n_nois
        self.n_hidden = hidden_units[0]
        self.layers = len(hidden_units)
        self.n_latent = n_latent
        self.rnn_model = rnn_model
        self.vae_as_input = vae_as_input

        # RNN part
        # batch_first=True to set (batch, seq, feature)
        # default is (seq_len, batch, input_size)
        rnn_models = {
            'rnn': torch.nn.RNN,
            'gru': torch.nn.GRU,
            'lstm': torch.nn.LSTM,
            }
        self.encoder = rnn_models[rnn_model](
            n_nois + n_feats,  # it receives both ramps and f0s
            self.n_hidden,
            batch_first=False,
            num_layers=self.layers
            )
        self.decoder = rnn_models[rnn_model](
            n_nois,
            self.n_hidden,
            batch_first=False,
            num_layers=self.layers
            )
        self.out = torch.nn.Linear(self.n_hidden, n_feats)

        # VAE part
        self.tanh = torch.tanh
        self.enc_mu = torch.nn.Linear(self.n_hidden, n_latent)
        self.enc_logvar = torch.nn.Linear(self.n_hidden, n_latent)
        if vae_as_input:
            self.lat_to_input = torch.nn.Linear(n_latent, n_nois)
        else:
            self.lat_to_input = torch.nn.Linear(n_latent, self.n_hidden)

        self.reset_parameters()

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.decoder.reset_parameters()
        self.out.reset_parameters()

        self.enc_mu.reset_parameters()
        self.enc_logvar.reset_parameters()
        self.lat_to_input.reset_parameters()

    def sample(self, mu, logvar):
        std = torch.exp(.5 * logvar)
        eps = torch.zeros_like(logvar).normal_()
        return mu + eps*std

    def init_zeros(self, tensor):
        # h0s = tensor.new_zeros((1, self.n_hidden))
        h0s = tensor.new_zeros(tensor.size())
        return h0s

    def encode(
            self,
            nois_f0s
            ):
        if self.rnn_model == 'lstm':
            __, (h_encoder, __) = self.encoder(nois_f0s)
        else:
            __, h_encoder = self.encoder(nois_f0s)
        return h_encoder

    def decode(
            self,
            z,
            nois,
            lens,
            max_len
            ):
        if self.vae_as_input:
            x = self.lat_to_input(z)
            # print(f'tensor size x {x.size()}')
            # x = x.unsqueeze(dim=0)  # for the seq size
            if self.rnn_model == 'lstm':
                __, (hx, __) = self.decoder(x)  # hx, cx default to 0
            else:
                __, hx = self.decoder(x)
        else:  # vae as hidden
            hx = self.lat_to_input(z)
            # hx = hx.unsqueeze(dim=0)
        # if lens[0] == 3:
        #     print(f'hx {hx}')
        # print(f'tensor size hx {hx.size()}')

        # print(f'tensor size nois {nois.size()}')
        # if lens[0] == 3:
        #     print(f'nois {nois}')
        nois = rnn_utils.pack_padded_sequence(
            nois, lens
            )
        # if lens[0] == 3:
        #     print(f'nois packed {nois}')
        # print(f'tensor size nois {nois.data.size()}')
        if self.rnn_model == 'lstm':
            cx = self.init_zeros(hx)
            hos, (__, __) = self.decoder(nois, (hx, cx))
        else:
            hos, __ = self.decoder(nois, hx)
        # if lens[0] == 3:
        #     print(f'hos {hos}')
        # print(f'tensor size ho {hos.data.size()}')
        # y = hos.copy()
        hos, __ = rnn_utils.pad_packed_sequence(
            hos, padding_value=0.0, total_length=max_len,
            )
        # if lens[0] == 3:
        #     print(f'hos {hos}')
        y = self.out(hos)
        # print(f'tensor size y {y.data.size()}')
        y = rnn_utils.pack_padded_sequence(
            y, lens,
            )
        return y

    def forward(
            self,
            nois_f0s,  # seq_len x n_samples x n_nois
            nois,  # seq_len x n_samples x n_nois
            lens,
            max_len,
            ):
        nois_f0s = rnn_utils.pack_padded_sequence(
            nois_f0s, lens
            )
        # if lens[0] == 3:
        #     print(f'nois_f0s {nois_f0s}')
        #     print(f'nois {nois}')
        # encoder
        # print(f'tensor size nois_fs {nois_f0s.size()}')
        h_encoder = self.encode(nois_f0s)
        # if lens[0] == 3:
        #     print(f'h_encoder {h_encoder}')
        # print(f'tensor size h_encoder {h_encoder.size()}')

        # latent space
        mu = self.enc_mu(h_encoder)
        # print(f'tensor size mu {mu.size()}')
        logvar = self.enc_logvar(h_encoder)
        # print(f'tensor size mu {logvar.size()}')
        if self.training:  # sample only on training
            z = self.sample(mu, logvar)
        else:
            z = mu
        # if lens[0] == 3:
        #     print(f'z {z}')
        # print(f'tensor size z {z.size()}')

        # decoder
        y = self.decode(z, nois, lens, max_len)
        # if lens[0] == 3:
        #     print(f'y {y}')
        return y, z, mu, logvar


class VAEModel(torch.nn.Module):
    """Define VAE network.

    This is a VAE contour generator that learns a latent space representation
    of the variety in a prosodic contours' realisation based on the contours
    themselves.
    """

    def __init__(
            self,
            n_contour,
            enc_n_hiddens,
            n_latent,
            dec_n_hiddens=None
            ):
        super().__init__()
        self.activation = torch.tanh

        if type(enc_n_hiddens) != list:
            enc_n_hiddens = [enc_n_hiddens]  # in case it is not a list already
        self.enc_n_hiddens = enc_n_hiddens

        if dec_n_hiddens is None:
            dec_n_hiddens = enc_n_hiddens
        self.dec_n_hiddens = dec_n_hiddens

        # contour generator network
        n_last = n_contour
        for i, n_hidden in enumerate(enc_n_hiddens):
            setattr(
                self, f'enc_hidden{i}', torch.nn.Linear(n_last, n_hidden)
                )
            n_last = n_hidden
        self.enc_mu = torch.nn.Linear(n_last, n_latent)
        self.enc_logvar = torch.nn.Linear(n_last, n_latent)

        n_last = n_latent
        for i, n_hidden in enumerate(dec_n_hiddens):
            setattr(
                self, f'dec_hidden{i}', torch.nn.Linear(n_last, n_hidden)
                )
            n_last = n_hidden
        self.dec_out = torch.nn.Linear(n_last, n_contour)

        self.reset_parameters()

    def reset_parameters(self):
        """Reset model parameters."""
        for i in range(len(self.enc_n_hiddens)):
            getattr(self, f'enc_hidden{i}').reset_parameters()
        for i in range(len(self.dec_n_hiddens)):
            getattr(self, f'dec_hidden{i}').reset_parameters()
        self.enc_mu.reset_parameters()
        self.enc_logvar.reset_parameters()
        self.dec_out.reset_parameters()

    def encode(self, x):
        """Encode contour to latent space distribution."""
        for i in range(len(self.enc_n_hiddens)):
            layer = getattr(self, f'enc_hidden{i}')
            x = layer(x)
            x = self.activation(x)
        mu = self.enc_mu(x)
        logvar = self.enc_logvar(x)
        return mu, logvar

    def sample(self, mu, logvar):
        """Sample the latent space."""
        std = torch.exp(.5 * logvar)
        eps = torch.zeros_like(logvar).normal_()
        return mu + eps*std

    def decode(self, z):
        """Decode sample to contour."""
        y = z
        for i in range(len(self.dec_n_hiddens)):
            layer = getattr(self, f'dec_hidden{i}')
            y = layer(y)
            y = self.activation(y)
        y = self.dec_out(y)
        return y

    def forward(self, x):
        """Pass sample through model."""
        mu, logvar = self.encode(x)
        if self.training:  # sample only on training
            z = self.sample(mu, logvar)
        else:
            z = mu
        y = self.decode(z)
        return y, z, mu, logvar


class VAEWrapper():
    """Integrates the VAE network in a model that includes fit and predict.

    The model wrapper is used to initialise the VAE network, and includes fit
    and predict methods.
    """

    def __init__(self, params=None):

        self.model_type = params.model_type
        self.rnn_model = params.rnn_model
        self.n_feats = params.n_feats
        self.n_nois = params.n_nois
        self.hidden_units = params.hidden_units
        self.n_latent = params.n_latent

        self.reg_type = params.reg_type
        self.reg_vae = params.reg_vae
        self.l2 = params.l2

        self.batch_size = params.batch_size
        self.shuffle = params.shuffle
        self.seed = params.seed
        torch.manual_seed(params.seed)

        self.optimizer_type = params.optimizer_type
        self.learn_rate = params.learn_rate

        self.use_cuda = params.use_cuda
        self.verbose = params.verbose

        self.init_model()

    def init_model(self):
        """Initialize model."""
        # init model
        if self.model_type == 'dnn':
            self.model = VAEModel(
                    n_contour=self.n_feats,
                    enc_n_hiddens=self.hidden_units,  # from list to int
                    n_latent=self.n_latent,
                    dec_n_hiddens=None,  # automatically equals enc hiddens
                    )
        else:
            self.model = RVAEModel(
                n_feats=self.n_feats,
                n_nois=self.n_nois,
                hidden_units=self.hidden_units,
                n_latent=self.n_latent,
                rnn_model=self.rnn_model,
                vae_as_input=False,
                )

        self.criterion = torch.nn.MSELoss()

        if self.use_cuda:
            self.model.cuda()
            self.criterion.cuda()

        self.init_optimizer()

    def init_optimizer(self):
        """Initialize optimizer."""
        if self.optimizer_type == 'adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.learn_rate,
                weight_decay=self.l2
                )
        else:
            raise NotImplementedError('Optimizer not implemented!')

    def reset_model(self):
        """Reset model parameters and optimizer."""
        self.model.reset_parameters()
        self.init_optimizer()

    def fit(
            self,
            x_train,  # f0s
            x_nois=None,  # for the RVAE model
            x_lens=None,  # for the RVAE model - used for packing
            x_val=None,  # no validation for this application scenario
            x_nois_val=None,  # for RVAE
            x_lens_val=None,  # for RVAE
            ):
        """Fit model to data."""
        # set to training - uses sampling from latent space
        self.model.train()

        # if x_train.ndim == 1:  # if it's a single sample - maybe obsolete?
        #     x_train = np.expand_dims(x_train, 0)
        #     if x_decoder is not None:
        #         x_decoder = np.expand_dims(x_decoder, 0)

        if self.model_type == 'dnn':
            n_samples = x_train.shape[0]
            if x_val is not None:
                n_samples_val = x_val.shape[0]
        else:
            max_len = x_train.shape[0]  # seq_len x samples x features
            n_samples = x_train.shape[1]  # seq_len x samples x features
            if x_val is not None:
                n_samples_val = x_val.shape[1]

        is_reg = (self.reg_vae > 0) and (self.reg_type in ['mmd', 'kld'])

        # cast to Tensor and upload to gpu
        dtype = torch.float32
        if self.use_cuda:
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        x_train = torch.from_numpy(x_train).type(dtype).to(device)


        if x_nois is not None:
            x_nois = torch.from_numpy(x_nois).type(dtype).to(device)
        if x_lens is not None:
            # required on cpu for pad packed seq
            x_lens = torch.from_numpy(x_lens).type(dtype).to('cpu')

        if x_val is not None:
            x_val = torch.from_numpy(x_val).type(dtype).to(device)
        if x_nois_val is not None:
            x_nois_val = torch.from_numpy(x_nois_val).type(dtype).to(device)
        if x_lens_val is not None:
            # required on cpu for pad packed seq
            x_lens_val = torch.from_numpy(x_lens_val).type(dtype).to('cpu')

        if self.batch_size == 'auto':
            batch_size = np.min([64, n_samples])
        elif self.batch_size == 'all':
            batch_size = n_samples
        else:
            batch_size = int(np.min((int(self.batch_size), n_samples)))
        if self.shuffle:
            sampler = RandomSampler(range(n_samples))
        else:
            sampler = SequentialSampler(range(n_samples))

        # batch loop
        batch_iter = BatchSampler(
            sampler, batch_size=int(batch_size), drop_last=False
            )
        n_batches = len(batch_iter)
        loss_in_batch = np.zeros(n_batches)
        mse_in_batch = np.zeros(n_batches)
        if is_reg:
           reg_in_batch = np.zeros(n_batches)

        for i, batch_ind in enumerate(batch_iter):
            if self.verbose:
                print(f'\rtraining batch {i}/{n_batches} ', end='')
            if self.model_type == 'dnn':
                x_batch = x_train[batch_ind]
                prediction_batch, zs_batch, mus_batch, logvars_batch = \
                    self.model(x_batch)
                # get loss - must be (1. nn output, 2. target)
                mse = self.criterion(prediction_batch, x_batch)
            else:
                f0s_batch = x_train[:, batch_ind, :]
                nois_batch = x_nois[:, batch_ind, :]
                lens_batch = x_lens[batch_ind]
                nois_f0s_batch = torch.cat((nois_batch, f0s_batch), 2)

                prediction_packed, zs_batch, mus_batch, logvars_batch = \
                    self.model(
                        nois_f0s_batch, nois_batch, lens_batch, max_len
                        )

                f0s_packed = rnn_utils.pack_padded_sequence(
                    f0s_batch, lens_batch
                    )
                mse = self.criterion(prediction_packed.data, f0s_packed.data)
                if torch.isnan(mse):
                    raise ValueError('MSE is NaN!')

            loss = torch.zeros_like(mse)
            loss += mse  # otherwise it makes it the same reference

            if self.reg_vae > 0:
                if self.reg_type == 'mmd':
                    # print(f'tensor size zs_batch {zs_batch.size()}')
                    if self.model_type == 'rnn':
                        # z size is [1, 32, 2] - seq_len x n_batch x n_latent
                        # it should be:
                        # (batch_size, n_latent, n_modules) w/o the n_modules
                        zs_batch = zs_batch.squeeze(dim=0)
                    reg_loss = vae_utils.reg_loss(zs_batch)
                    # print(f' mmd {mmd}')
                    loss += self.reg_vae * reg_loss
                else:  # KLD
                    reg_loss = vae_utils.loss_kld(mus_batch, logvars_batch)
                    loss += self.reg_vae * reg_loss

            self.optimizer.zero_grad()  # clear gradients for next train
            # backpropagation, compute gradients
            loss.backward(retain_graph=True)
            self.optimizer.step()  # apply gradients

            loss_in_batch[i] = loss.item()
            mse_in_batch[i] = mse.item()
            if is_reg:
                reg_in_batch[i] = reg_loss.item()

        self.losses_ = loss_in_batch
        if any(np.isnan(self.losses_)):
            raise ValueError('Loss is NaN!')
        self.mses_ = mse_in_batch
        if is_reg:
            self.regs_ = reg_in_batch

        # evaluate on validation data
        if x_val is not None:
            with torch.no_grad():
                self.losses_train_ = self.losses_
                self.mses_train_ = self.mses_
                if is_reg:
                    self.regs_train_ = self.regs_

                # run batches
                sampler_val = SequentialSampler(range(n_samples_val))
                batch_iter = BatchSampler(
                    sampler_val, batch_size=int(batch_size), drop_last=False
                    )
                n_batches = len(batch_iter)

                loss_in_batch = np.zeros(n_batches)
                mse_in_batch = np.zeros(n_batches)
                if is_reg:
                   reg_in_batch = np.zeros(n_batches)

                self.model.eval()
                for i, batch_ind in enumerate(batch_iter):
                    if self.verbose:
                        print(f'\rvalidation batch {i}/{n_batches} ', end='')
                    if self.model_type == 'dnn':
                        x_batch = x_val[batch_ind]
                        prediction_batch, zs_batch, mus_batch, logvars_batch = \
                            self.model(x_batch)
                        mse = self.criterion(prediction_batch, x_batch)
                    else:
                        f0s_batch = x_val[:, batch_ind, :]
                        nois_batch = x_nois_val[:, batch_ind, :]
                        lens_batch = x_lens_val[batch_ind]
                        nois_f0s_batch = torch.cat((nois_batch, f0s_batch), 2)

                        prediction_packed, zs_batch, mus_batch, logvars_batch = \
                            self.model(
                                nois_f0s_batch, nois_batch, lens_batch, max_len
                                )
                        f0s_packed = rnn_utils.pack_padded_sequence(
                            f0s_batch, lens_batch
                            )
                        mse = self.criterion(prediction_packed.data, f0s_packed.data)
                        if torch.isnan(mse):
                            raise ValueError('MSE is NaN!')

                    loss = torch.zeros_like(mse)
                    loss += mse  # otherwise it makes it the same reference

                    if self.reg_vae > 0:
                        if self.reg_type == 'mmd':
                            # print(f'tensor size zs_batch {zs_batch.size()}')
                            if self.model_type == 'rnn':
                                # z size is [1, 32, 2] - seq_len x n_batch x n_latent
                                # it should be:
                                # (batch_size, n_latent, n_modules) w/o the n_modules
                                zs_batch = zs_batch.squeeze(dim=0)
                            reg_loss = vae_utils.reg_loss(zs_batch)
                            # print(f' mmd {mmd}')
                            loss += self.reg_vae * reg_loss
                        else:  # KLD
                            reg_loss = vae_utils.loss_kld(mus_batch, logvars_batch)
                            loss += self.reg_vae * reg_loss

                    loss_in_batch[i] = loss.item()
                    mse_in_batch[i] = mse.item()
                    if is_reg:
                       reg_in_batch[i] = reg_loss.item()

                self.losses_ = loss_in_batch
                if any(np.isnan(self.losses_)):
                    raise ValueError('Loss is NaN!')

                self.mses_ = mse_in_batch
                if is_reg:
                    self.regs_ = reg_in_batch

        if self.verbose:
            print('\r', end='')

    def predict(
            self,
            x,  # f0s
            x_nois=None,  # for the RVAE model
            x_lens=None,  # for the RVAE model - used for packing
            sample_vae=False
            ):
        """Make predictions for input data.

        x_train and masks are np arrays.
        sample_vae enables sampling of latent space.
        """
        if self.model_type == 'dnn':
            if x.ndim == 1:  # if it's a single sample
                x = np.expand_dims(x, 0)
            n_samples = x.shape[0]
            y = np.zeros((n_samples, self.n_feats))
        else:
            if x.ndim == 2:  # if it's a single sample
                x = np.expand_dims(x, 1)
            n_samples = x.shape[1]  # seq_len x samples x features
            max_len = x.shape[0]  # seq_len x samples x features
            y = np.zeros((max_len, n_samples, self.n_feats))

        # cast to Tensor
        dtype = torch.float32
        if self.use_cuda:
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        x = torch.from_numpy(x).type(dtype).to(device)

        if x_nois is not None:
            x_nois = torch.from_numpy(x_nois).type(dtype).to(device)
        if x_lens is not None:
            x_lens = torch.from_numpy(x_lens).type(dtype).to('cpu')
            # required on cpu for pad packed seq

        y = torch.from_numpy(y).type(dtype).to(device)

        batch_size = int(np.min((256, n_samples)))
        sampler = SequentialSampler(range(n_samples))

        if not sample_vae:
            self.model.eval()
        else:
            self.model.train()
        batch_iter = BatchSampler(
            sampler, batch_size=batch_size, drop_last=False
            )
        with torch.no_grad():
            for batch_ind in batch_iter:
                if self.model_type == 'dnn':
                    x_batch = x[batch_ind]
                    prediction_batch, *__ = self.model(x_batch)
                    y[batch_ind] = prediction_batch
                else:
                    # extract batches
                    f0s_batch = x[:, batch_ind, :]
                    nois_batch = x_nois[:, batch_ind, :]
                    lens_batch = x_lens[batch_ind]
                    nois_f0s_batch = torch.cat(
                        (nois_batch, f0s_batch), 2
                        )  # along axis 2

                    # get prediction
                    prediction_packed, *__ = self.model(
                        nois_f0s_batch, nois_batch, lens_batch, max_len
                        )

                    # upack prediction
                    prediction_batch, __ = rnn_utils.pad_packed_sequence(
                        prediction_packed,
                        padding_value=np.nan,
                        total_length=max_len,
                        )

                    y[:, batch_ind, :] = prediction_batch

        return y.detach().cpu().numpy()


def loss_mmd(zs):
    """Maximum Mean Discrepancy loss measure based on InfoVAEs.

    Code based on author's implementation:
    http://szhao.me/2017/06/10/a-tutorial-on-mmd-variational-autoencoders.html

    zs : torch Variable of dimensions (batch_size, n_latent, n_modules)
    mask : batch_size x n_modules
    """
    def compute_mmd(x, y):
        x_kernel = compute_kernel(x, x)
        y_kernel = compute_kernel(y, y)
        xy_kernel = compute_kernel(x, y)
        return (
            torch.mean(x_kernel)
            + torch.mean(y_kernel)
            - 2 * torch.mean(xy_kernel)
            )

    def compute_kernel(x, y):
        x_size = x.shape[0]  # 200
        y_size = y.shape[0]  # 200
        dim = x.shape[1]  # 2
        tiled_x = x.unsqueeze(dim=1).repeat(1, y_size, 1)
        tiled_y = y.unsqueeze(dim=0).repeat(x_size, 1, 1)
        return torch.exp(
            -torch.mean((tiled_x - tiled_y)**2, dim=2) / dim
            )

    true_samples = torch.randn_like(zs)
    return compute_mmd(true_samples, zs)


def loss_kld(mu, logvar):
    r"""Calculate KLD.

    adapted from https://github.com/pytorch/examples/tree/master/vae
    see Appendix B from VAE paper:
    Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    https://arxiv.org/abs/1312.6114

    :math:`\frac{1}{2} \cdot \sum(1 + log(\sigma^2) - \mu^2 - \sigma^2)`

    ``mu.shape = batch_size x n_latent x n_modules``

    ``mask = batch_size x n_modules``
    """
    if (logvar > 10).any():
        logvar.exp_()
    kld = -0.5 * torch.sum(
        1 + logvar - mu**2 - torch.exp(logvar), dim=0
        )
    kld = torch.mean(kld)
    return kld
