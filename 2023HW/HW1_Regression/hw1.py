# 2024/1/8
# zhangzhong

import math
import numpy as np
import pandas as pd
import os
import csv
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard.writer import SummaryWriter
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR


def same_seed(seed):
    '''Fixes random number generator seeds for reproducibility.'''
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_valid_split(data_set, valid_ratio, seed):
    '''Split provided training data into training set and validation set'''
    valid_set_size = int(valid_ratio * len(data_set))
    train_set_size = len(data_set) - valid_set_size
    train_set, valid_set = random_split(data_set, [
                                        train_set_size, valid_set_size], generator=torch.Generator().manual_seed(seed))
    return np.array(train_set), np.array(valid_set)


def predict(test_loader, model, device):
    model.eval()  # Set your model to evaluation mode.
    preds = []
    for x in tqdm(test_loader):
        x = x.to(device)
        with torch.no_grad():
            pred = model(x)
            preds.append(pred.detach().cpu())
    preds = torch.cat(preds, dim=0).numpy()
    return preds


class COVID19Dataset(Dataset):
    '''
    x: Features.
    y: Targets, if none, do prediction.
    '''

    def __init__(self, x, y=None):
        if y is None:
            self.y = y
        else:
            self.y = torch.FloatTensor(y)
        self.x = torch.FloatTensor(x)

    def __getitem__(self, idx):
        if self.y is None:
            return self.x[idx]
        else:
            return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)


class My_Model(nn.Module):
    def __init__(self, input_dim):
        super(My_Model, self).__init__()
        # TODO: modify model's structure, be aware of dimensions.
        self.layers = nn.Sequential(
            nn.LazyLinear(32),
            nn.LazyBatchNorm1d(),
            nn.ReLU(),
            # 这种少量的数据不适合dropout 也不适合太深的网络
            # 参数量太大也会导致过拟合
            # nn.Dropout(0.1),

            nn.LazyLinear(128),
            nn.LazyBatchNorm1d(),
            nn.ReLU(),
            # nn.Dropout(0.1),

            nn.LazyLinear(32),
            nn.LazyBatchNorm1d(),
            nn.ReLU(),
            # nn.Dropout(0.1),

            nn.LazyLinear(1)
        )

    def forward(self, x):
        x = self.layers(x)
        x = x.squeeze(1)  # (B, 1) -> (B)
        return x


def select_feat(train_data, valid_data, test_data, select_all=True):
    '''Selects useful features to perform regression'''

    # 怪不得怎么算都不对 test没给最后的positive rate呀 所以我在下面算的mse是不对的呀...
    # 所以y_test其实也没有用 这根本不是y呀
    # only choose the last column as labels
    y_train, y_valid, y_test = train_data[:, -
                                          1], valid_data[:, -1], test_data[:, -1]
    # choose all columns except the last one
    raw_x_train, raw_x_valid, raw_x_test = train_data[:,
                                                      :-1], valid_data[:, :-1], test_data

    if select_all:
        # select all feature columns
        feat_idx = list(range(raw_x_train.shape[1]))
    else:
        # 从35列开开始 到最后一列 因为raw已经去掉了原来的最后一列
        # feat_idx = [0,1,2,3,4] # TODO: Select suitable feature columns.
        feat_idx = list(range(35, raw_x_train.shape[1]))
        # try others feature idx
        # feat_idx = [34, 36, 51, 52, 54, 70, 72, 69]

    return raw_x_train[:, feat_idx], raw_x_valid[:, feat_idx], raw_x_test[:, feat_idx], y_train, y_valid, y_test


def trainer(train_loader, valid_loader, test_loader, model, config, device):

    # Define your loss function, do not modify this.
    criterion = nn.MSELoss(reduction='mean')

    # Define your optimization algorithm.
    # TODO: Please check https://pytorch.org/docs/stable/optim.html to get more available algorithms.
    # TODO: L2 regularization (optimizer(weight decay...) or implement by your self).
    # https://stackoverflow.com/questions/42704283/l1-l2-regularization-in-pytorch
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config['learning_rate'], weight_decay=0.01)
    writer = SummaryWriter()  # Writer of tensoboard.

    if not os.path.isdir('./models'):
        os.mkdir('./models')  # Create directory of saving models.

    n_epochs, best_loss, step, early_stop_count = config['n_epochs'], math.inf, 0, 0

    # lr: float = 0.001
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=0.001)
    # warmup_epochs = 100
    # num_epochs = 3000
    # scheduler = SequentialLR(
    #     optimizer=optimizer,
    #     schedulers=[
    #         LinearLR(optimizer=optimizer, start_factor=0.1,
    #                  total_iters=warmup_epochs),
    #         CosineAnnealingLR(optimizer=optimizer,
    #                           T_max=num_epochs-warmup_epochs)
    #     ],
    #     milestones=[warmup_epochs]
    # )

    for epoch in range(n_epochs):
        model.train()  # Set your model to train mode.
        loss_record = []

        # tqdm is a package to visualize your training progress.
        train_pbar = tqdm(train_loader, position=0, leave=True)

        for x, y in train_pbar:
            optimizer.zero_grad()               # Set gradient to zero.
            x, y = x.to(device), y.to(device)   # Move your data to device.
            pred = model(x)
            loss = criterion(pred, y)
            # Compute gradient(backpropagation).
            loss.backward()
            optimizer.step()                    # Update parameters.
            step += 1
            loss_record.append(loss.detach().item())

            # Display current epoch number and loss on tqdm progress bar.
            train_pbar.set_description(f'Epoch [{epoch+1}/{n_epochs}]')
            train_pbar.set_postfix({'loss': loss.detach().item()})

            # update lr
            # scheduler.step()

        mean_train_loss = sum(loss_record)/len(loss_record)
        writer.add_scalar('Loss/train', mean_train_loss, step)

        model.eval()  # Set your model to evaluation mode.
        loss_record = []
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)
                loss = criterion(pred, y)

            loss_record.append(loss.item())

        mean_valid_loss = sum(loss_record)/len(loss_record)
        print(
            f'Epoch [{epoch+1}/{n_epochs}]: Train loss: {mean_train_loss:.4f}, Valid loss: {mean_valid_loss:.4f}')
        writer.add_scalar('Loss/valid', mean_valid_loss, step)

        # model.eval() # Set your model to evaluation mode.
        # loss_record = []
        # for x, y in test_loader:
        #     x, y = x.to(device), y.to(device)
        #     with torch.no_grad():
        #         pred = model(x)
        #         loss = criterion(pred, y)

        #     loss_record.append(loss.item())

        # mean_test_loss = sum(loss_record)/len(loss_record)
        # print(f'Epoch [{epoch+1}/{n_epochs}]: Test loss: {mean_test_loss:.4f}')
        # writer.add_scalar('Loss/test', mean_test_loss, step)

        if mean_valid_loss < best_loss:
            best_loss = mean_valid_loss
            # Save your best model
            torch.save(model.state_dict(), config['save_path'])
            print('Saving model with loss {:.3f}...'.format(best_loss))
            early_stop_count = 0
        else:
            early_stop_count += 1

        if early_stop_count >= config['early_stop']:
            print('\nModel is not improving, so we halt the training session.')
            return


# device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
config = {
    'seed': 42,      # Your seed number, you can pick your lucky number. :)
    'select_all': False,   # Whether to use all features.
    'valid_ratio': 0.2,   # validation_size = train_size * valid_ratio
    'n_epochs': 3000,     # Number of epochs.
    'batch_size': 32,       # 降低batchsize是真的有用 感觉降到32可能就过boss baseline了 不过训练时间太长了 还是不玩了
    # 换成 cosine learning scheduler
    'learning_rate': 1e-3,
    # If model has not improved for this many consecutive epochs, stop training.
    'early_stop': 600,
    'save_path': './models/model.ckpt'  # Your model will be saved here.
}


# same_seed(config['seed'])
# to_numpy ndarray
train_data, test_data = pd.read_csv(
    './covid_train.csv').values, pd.read_csv('./covid_test.csv').values
train_data, valid_data = train_valid_split(
    train_data, config['valid_ratio'], config['seed'])

# Print out the data size.
print(f"""train_data size: {train_data.shape} 
valid_data size: {valid_data.shape} 
test_data size: {test_data.shape}""")

# Select features
x_train, x_valid, x_test, y_train, y_valid, y_test = select_feat(
    train_data, valid_data, test_data, config['select_all'])

# Print out the number of features.
print(f'number of features: {x_train.shape[1]}')

# 这里修改一下 让test_loader也可以拿到y_test
train_dataset, valid_dataset, test_dataset = COVID19Dataset(x_train, y_train), \
    COVID19Dataset(x_valid, y_valid), \
    COVID19Dataset(x_test)

# Pytorch data loader loads pytorch dataset into batches.
train_loader = DataLoader(
    train_dataset, batch_size=config['batch_size'], shuffle=True)
valid_loader = DataLoader(
    valid_dataset, batch_size=config['batch_size'], shuffle=True)
test_loader = DataLoader(
    test_dataset, batch_size=config['batch_size'], shuffle=False)

print(y_test)


# put your model and data on the same computation device.
model = My_Model(input_dim=x_train.shape[1]).to(device)
trainer(train_loader, valid_loader, test_loader, model, config, device)


def save_pred(preds, file):
    ''' Save predictions to specified file '''
    with open(file, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['id', 'tested_positive'])
        for i, p in enumerate(preds):
            writer.writerow([i, p])


model = My_Model(input_dim=x_train.shape[1]).to(device)
model.load_state_dict(torch.load(config['save_path']))
preds = predict(test_loader, model, device)
save_pred(preds, 'pred.csv')


# 对于这种小型的数据集，应该使用小型的网络
# 尤其是对于线性回归问题，可以画出各个feature对于y的散点图，肉眼观察相关性来查找
# dropout往往没用 而 batchnorm比较有用
# 使用 cosine scheduler 导致波浪形的loss曲线 而adam就没有 ？？这很奇怪
