#!/usr/bin/env python3
"""
VAE training utility functions for Data spread calculation.

@author:
    Branislav Gerazov Apr 2020
"""
import numpy as np
try:
    import torch
    from data_spread_tool import data_vae_models as vae_models
except ImportError:
    print('No PyTorch installed!')
try:
    from sklearn.model_selection import ShuffleSplit, train_test_split
except ImportError:
    print('No sklearn installed!')


def init_model(params):
    """Init model wrapper."""
    # init wrapper
    print('Initialising VAE wrapper ...')
    if params.seed is not None:
        torch.manual_seed(params.seed)
    wrapper = vae_models.VAEWrapper(params)
    return wrapper


def train_model(
        data,
        wrapper,
        data_nois=None,  # for the RVAE
        data_lens=None,  # for the RVAE
        params=None
        ):
    """Train deep Variational Autoencoder wrapper based on data."""
    # training params
    # model_type = params.model_type
    max_iter = params.max_iter
    use_cuda = params.use_cuda
    early_stopping = params.early_stopping
    early_thresh = params.early_thresh
    patience = params.patience
    use_validation = params.use_validation
    validation_size = params.validation_size
    seed = params.seed
    # np.random.seed()  # best practice has changed
    # https://numpy.org/doc/stable/reference/random/generated/numpy.random.seed.html
    # torch.random.seed(seed)
    # torch.random.set_rng_state(seed)

    # wrapper = init_model(params)

    # do the epochs
    print('Training VAE model ...')
    batch_losses = np.asarray([])
    batch_mses = np.asarray([])
    epoch_losses = np.asarray([])
    epoch_mses = np.asarray([])
    epoch_cnt = np.asarray([])
    print('='*42)
    best_error = np.inf
    print(f'Data shape {data.shape}')
    # train-validation split
    if use_validation:  # make a train-val split
        print('Splitting data into train-val sets ...')
        # if params.shuffle:
        # ss = ShuffleSplit(
        #     test_size=validation_size,
        #     random_state=seed,
        #     )
        if params.model_type == 'dnn':
            n_samples = data.shape[0]  # n_samples, n_feats (contour length)
        elif params.model_type == 'rnn':
            n_samples = data.shape[1]  # n_nois, n_samples, n_feats
        else:
            raise ValueError('Model not recognized!')
        val_ind = np.random.random_sample((n_samples, )) < validation_size
        train_ind = ~val_ind
        if params.model_type == 'dnn':
            x_train, x_val = data[train_ind], data[val_ind]
        elif params.model_type == 'rnn':
            x_train, x_val = data[:, train_ind, :], data[:, val_ind, :]
            if data_nois is not None:
                # shape is n_nois, n_samples, n_ramps
                x_nois, x_nois_val = data_nois[:, train_ind, :], data_nois[:, val_ind, :]
            if data_lens is not None:
                x_lens, x_lens_val = data_lens[train_ind], data_lens[val_ind]
        print(
            f'Training data length {sum(train_ind)}, '
            f'validation data length {sum(val_ind)}.'
            )
    else:
        x_train = data
        x_nois = data_nois
        x_lens = data_lens
        x_val = None
        x_nois_val = None
        x_lens_val = None

    is_reg = (params.reg_vae > 0) and (params.reg_type in ['mmd', 'kld'])

    for iteration in range(0, max_iter):
        print(f'Epoch {iteration+1}/{max_iter} ...')

        # this only does one epoch
        wrapper.fit(
            x_train,
            x_nois=x_nois,  # for the RVAE
            x_lens=x_lens,  # for the RVAE
            x_val=x_val,
            x_nois_val=x_nois_val,  # for RVAE
            x_lens_val=x_lens_val,  # for RVAE
            )

        # accumulate losses for epoch
        mse_epoch = wrapper.mses_.mean()
        loss_epoch = wrapper.losses_.mean()
        if is_reg:
            reg_loss = wrapper.regs_.mean()
        batch_mses = np.r_[batch_mses, wrapper.mses_]
        batch_losses = np.r_[batch_losses, wrapper.losses_]
        epoch_mses = np.r_[epoch_mses, mse_epoch]
        epoch_losses = np.r_[epoch_losses, loss_epoch]
        if iteration == 0:
            epoch_cnt = np.array([wrapper.mses_.size])
        else:
            epoch_cnt = np.r_[epoch_cnt, epoch_cnt[-1] + wrapper.mses_.size]

        if is_reg:
            print(
                f'mse: {mse_epoch:.5f} \t '
                f'{params.reg_type}: {reg_loss:.5f} \t '
                f'loss: {loss_epoch:.5f}'
                )
        else:
            print(
                f'mse: {mse_epoch:.5f} \t loss: {loss_epoch:.5f}'
                )

        if early_stopping:  # check if error is not improving
            if best_error - loss_epoch > early_thresh:
                patience = params.patience
            else:
                patience -= 1
                print(
                    f'loosing patience: {patience} '
                    f'best loss: {best_error}'
                    )

        if best_error > loss_epoch:  # update best error
            best_error = loss_epoch
            # # run a forward pass on all the data to get the predictions
            # y_pred = wrapper.predict(
            #     x_all,
            #     x_nois=data_nois,  # for the RVAE
            #     x_lens=data_lens,  # for the RVAE
            #     )
            # presave wrapper
            # best_output = y_pred
            best_epoch = iteration
            best_losses = (
                batch_losses, epoch_cnt, epoch_losses, batch_mses, epoch_mses
                )
            best_model = wrapper.model.state_dict()
            best_optim = wrapper.optimizer.state_dict()  # for cont training
            best_return = best_model, best_optim, best_losses, best_error

        # stop if criteria are met:
        # last iteration or we've run out of patience
        if (iteration == max_iter - 1) or not patience:
            # y_pred = best_output
            iteration = best_epoch
            if early_stopping:
                print(
                    'Best error at '
                    'epoch {} : mse {:.5f} \t loss {:.5f}'.format(
                        best_epoch, best_losses[4][-1], best_losses[2][-1]
                        )
                    )
            break  # for the patience to work

    # return
    best_model, best_optim, best_losses, best_error = best_return
    wrapper.model.load_state_dict(best_model)
    wrapper.optimizer.load_state_dict(best_optim)
    batch_losses, epoch_cnt, epoch_losses, batch_mses, epoch_mses = best_losses
    if use_cuda:
        wrapper.model.cpu()
        wrapper.criterion.cpu()
    return wrapper, best_error, batch_losses, epoch_cnt, epoch_losses


def get_mus(
        data,
        model,
        ):
    """Obtain mus from latent space mapping of data."""
    mus = []
    sigmas = []
    for sample in data:
        dtype = torch.FloatTensor
        x = torch.tensor(sample).type(dtype)
        mu, logvar = model.encode(x)
        mu = mu.detach().cpu().numpy()
        mus.append(mu)
        sigma = torch.exp(logvar/2)
        sigma = sigma.detach().cpu().numpy()
        sigmas.append(sigma)
    return np.array(mus), np.array(sigmas)


def get_mus_rvae(
        data_f0s,
        data_nois=None,  # for the RVAE
        data_lens=None,  # for the RVAE
        model=None,
        ):
    """Obtain mus from latent space mapping of data."""
    max_len = int(np.max(data_lens))
    mus = []
    sigmas = []
    for i in range(data_f0s.shape[1]):
        length = [data_lens[i]]
        dtype = torch.FloatTensor
        f0s = torch.tensor(
            data_f0s[:length[0], i, :]
            ).type(dtype).unsqueeze(1)
        nois = torch.tensor(
            data_nois[:length[0], i, :]
            ).type(dtype).unsqueeze(1)
        nois_f0s = torch.cat((nois, f0s), 2)  # along axis 2
        __, __, mu, logvar = model(nois_f0s, nois, length, max_len)
        mu = mu.detach().cpu().numpy().squeeze(1)
        mus.append(mu)
        sigma = torch.exp(logvar.squeeze(1) / 2)
        sigma = sigma.detach().cpu().numpy()
        sigmas.append(sigma)
    return np.array(mus), np.array(sigmas)


def get_noi_ramps(seq_len):
    nois = []
    for i in range(seq_len):
        nois.append([i, seq_len - i - 1])
    return np.array(nois, dtype='float32')
