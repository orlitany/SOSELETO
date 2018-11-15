# coding: utf-8

# Prepare data for the region proposal network train
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets


def fetch_mnist_data(params, data_path, labels, num_shot):
    '''Prepares mnist test data loader and small target set for training.
    Args:
        params (dict): containing: batch_size, random seed
        data_path: (string) path to where mnist data is stored \ will be downloaded to
        labels: (list) the desired subset of labels (e.g. [0, 1, 2])
        num-shot: (int) how many examples are allowed in the few shot setting    
    Returns:
        target set (tuple of Tensors): data and label for num_shot times examples per label
        mnist_test_loader (dataloader): modified dataloader with just the partial labels
    '''
    
    # add reproducibility of the target set
    if params['random_seed'] is not None:
        np.random.seed(params['random_seed'])
    
    mnist_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    # load mnist train loader
    mnist_train = datasets.MNIST(root=data_path, download=True, train=True, transform=mnist_transform)
    mnist_train_loader = torch.utils.data.DataLoader(dataset=mnist_train,
                                                     batch_size=params['batch_size'],
                                                     shuffle=False,
                                                     num_workers=1)
    # randomly choose a few examples from each label
    mnist_target_set = []
    mnist_target_labels = []
    for lbl in labels:
        sample_inds = np.random.permutation(np.where(mnist_train.train_labels == lbl)[0])[:num_shot]        
        for si in sample_inds:
            mnist_target_set.append(mnist_train_loader.dataset.__getitem__(si)[0])        
        mnist_target_labels += num_shot*[lbl]
        
    mnist_target_set = torch.cat(mnist_target_set).unsqueeze(1)
    mnist_target_labels = torch.tensor(mnist_target_labels)

    # Prepare MNIST test
    mnist_test = datasets.MNIST(root=data_path, download=True, train=False, transform=mnist_transform)
    label_inds = np.where(np.in1d(mnist_test.test_labels, labels))[0]
    mnist_test.test_labels = mnist_test.test_labels[label_inds]
    mnist_test.test_data = mnist_test.test_data[label_inds]

    mnist_test_loader = torch.utils.data.DataLoader(dataset=mnist_test,
                                                     batch_size=params['batch_size'],
                                                     shuffle=False,
                                                     num_workers=1)
    
    
    
    return (mnist_target_set, mnist_target_labels), mnist_test_loader


def fetch_svhn_data(params, data_path, labels):
    '''Prepares mnist test data loader and small target set for training.
    Args:
        params (dict): containing: batch_size, random seed
        data_path: (string) path to where mnist data is stored \ will be downloaded to
        labels: (list) the desired subset of labels (e.g. [0, 1, 2])
        num-shot: (int) how many examples are allowed in the few shot setting    
    Returns:        
        svhn_data_loader (dataloader): modified dataloader with just the partial labels
    '''

    svhn_transform = transforms.Compose([
        transforms.Scale([28, 28]), 
        transforms.RandomHorizontalFlip(),
        transforms.Grayscale(num_output_channels=1),                    
        transforms.ToTensor(),
        transforms.Normalize([0.444], [0.175])
    ])

    # manipulate svhn dataset to keep only relevant labels
    svhn_train = datasets.SVHN(root=data_path, download=True, transform=svhn_transform, split='train')
    keep_ind = np.where(np.isin(svhn_train.labels, SVHN_LABELS))
    svhn_train.labels = svhn_train.labels[keep_ind]
    svhn_train.data = svhn_train.data[keep_ind]
    
    svhn_train_loader = torch.utils.data.DataLoader(dataset=svhn_train,
                                              batch_size=params['batch_size'],
                                              shuffle=True,
                                              num_workers=1)
    return svhn_train_loader

#############################
# TEST FUNCTIONS
#############################
def test_fetch_mnist_data():
    # TODO: add more tests
    data_path = './../data/'
    labels = [0, 1, 8]
    num_shot = 7
    (mnist_target_set, mnist_target_labels), mnist_test_loader = fetch_mnist_data(
        {'batch_size': 8, 'random_seed': 42}, data_path, labels, num_shot)

    # check output types
    assert type(mnist_target_set) == torch.Tensor, 'bad type'
    assert type(mnist_target_labels) == torch.Tensor, 'bad type'
    assert type(mnist_test_loader) == torch.utils.data.dataloader.DataLoader, 'bad type'

    # check dimensions
    assert mnist_target_set.shape[0] == num_shot * len(labels), 'wrong number of target instances'    
    return 'fetch mnist data test passed'
    
def test_fetch_svhn_data():
    # TODO: add more tests
    data_path = './../data/'
    labels = [5, 6, 7, 8]    
    svhn_train_loader = fetch_svhn_data(
        {'batch_size': 4}, data_path, labels)

    # check output types    
    assert type(svhn_train_loader) == torch.utils.data.dataloader.DataLoader, 'bad type'

        
    return 'fetch svhn data test passed'

if __name__ == '__main__':
    print('Running some tests...')
    print(test_fetch_mnist_data())
    
    