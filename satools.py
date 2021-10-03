"""
This is a set of helper functions for ATLAS-collective-AE.ipynb
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from fastai.data import core

def normalize(data):
    """
    Normalize data before AE encoding. The data is assumed to be in the column order of [pt, eta, phi, E].
    
    pt and E are assumed to be given in MeV
    
    Normalization scheme is log10([GeV]) for pt and E, /3 for eta, phi
    """
    outdata = data.copy()
    outdata[:, [0, 3]] = np.log10(outdata[:, [0, 3]] * 1e-3)
    outdata[:, [1, 2]] = outdata[:, [1, 2]] / 3
    return outdata

def unnormalize(data):
    """
    Unnormalize data after AE decoding, to recover initial magnitudes. Returns pt and E in [GeV].
    """
    outdata = data.copy()
    outdata[:, [0, 3]] = 10**outdata[:, [0, 3]]
    outdata[:, [1, 2]] = outdata[:, [1, 2]] * 3
    return outdata

def make_DataLoaders(train, test, batch_size=256, shuffle=True):
    """
    Bundle training and testing data in DataLoaders from torch.utils.data
    
    Since this is intended to be used with an autoencoder, training input is the same as training target. Same goes for testing.
    
    Input needs to be numpy.array or torch.tensor
    """
    if type(train) != torch.Tensor:
        train = torch.tensor(train, dtype=torch.float)
    if type(test) != torch.Tensor:
        test = torch.tensor(test, dtype=torch.float)
    train_x = train
    train_y = train_x
    train_ds = TensorDataset(train_x, train_y)
    
    test_x = test
    test_y = test_x
    test_ds = TensorDataset(test_x, test_y)
    
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle)
    test_dl = DataLoader(test_ds, batch_size=batch_size)
    dls = core.DataLoaders(train_dl, test_dl)
    return dls

def group(data, group_size=2, n_features=3):
    """
    Create groups by concatenating a number of consecutive rows equal to group_size.
    """
    if group_size == 1:
        print('WARNING! Group size is = 1. Is this intentional?')
    data_rem = len(data) % group_size
    if not data_rem == 0:
        print(f'Number of rows in data not divisible by group size {group_size}. Truncating. Size before truncation: {len(data)}')
        data = data[:-data_rem]
        print(f'After truncation: {len(data)}')
    data_cat = data[::group_size]
    for i in range(1, group_size):
        if isinstance(data, np.ndarray):
            data_cat = np.hstack([data_cat, data[i::group_size]])
        else:
            data_cat = torch.hstack([data_cat, data[i::group_size]])
    return data_cat

def ungroup(pred, group_size=2, n_features=3):
    """
    Separate grouped data into individual instances.
    """
    if group_size == 1:
        print('WARNING! Group size is = 1. Is this intentional?')
    # Calculate shape of ungrouped data
    num_rows = int(pred.shape[0] * group_size)
    num_cols = int(pred.shape[1] / group_size)
    assert pred.shape[1] % group_size == 0, 'Group size {} does not go evenly into input number of columns {}. Has the data been altered after decoding?'.format(group_size, pred.shape[1])
    
    # Ungroup
    if isinstance(pred, np.ndarray):
        ungrouped_pred = np.zeros([num_rows, num_cols])
    else:
        ungrouped_pred = torch.zeros([num_rows, num_cols])
    for i in range(group_size):
        ungrouped_pred[i::group_size, :] = pred[:, i*num_cols:i*num_cols + num_cols]
    return ungrouped_pred