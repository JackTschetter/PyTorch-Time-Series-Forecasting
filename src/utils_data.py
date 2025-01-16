import torch
import numpy as np
import pandas as pd
from torch.autograd import Variable
from scipy.stats import pearsonr


class Data_utility(object):

    def __init__(self, file_name, train, valid, device, pred_window, hist_window, normalize=2):
        self.rawdat, self.rawdat_df = self.load_data(file_name)
        self.device = device
        self.hist_window = hist_window
        self.pred_window = pred_window
        self.dat = np.zeros(self.rawdat.shape)
        self.n, self.m = self.dat.shape  # (time, # of sensors)
        self._normalized(normalize)
        self._split(int(train * self.n), int((train + valid) * self.n))

        self.scale = torch.ones(self.m).to(device)

    def load_data(self, file_name):
        file_name = "data_" + file_name + ".csv"
        data_df = pd.read_csv(file_name)
        if "wiki_small" in file_name:
            data_df["date"] = pd.to_datetime(data_df["date"])
            data_df = data_df.rename(columns={"date": "ds"})
            data_arr = np.array(data_df[["dm", "ml", "nn", "py"]])
        elif "electricity" in file_name:
            data_df["date"] = pd.to_datetime(data_df["date"])
            NUM_COL = 322  # 322
            data_df = data_df.rename(columns={"date": "ds"}).iloc[:, range(0, NUM_COL)]
            data_arr = np.array(data_df.iloc[:, range(1, NUM_COL)])
        elif "traffic" in file_name:
            data_df["date"] = pd.to_datetime(data_df["date"])
            NUM_COL = 862  # 862
            data_df = data_df.rename(columns={"date": "ds"}).iloc[:, range(0, NUM_COL)]
            data_arr = np.array(data_df.iloc[:, range(1, NUM_COL)])

        return data_arr, data_df

    def _normalized(self, normalize):
        # normalized by the maximum value of entire matrix.

        if normalize == 0:
            self.dat = self.rawdat

        if normalize == 1:
            self.dat = self.rawdat / np.max(self.rawdat)
            self.dat_df = self.rawdat_df
            self.dat_df.iloc[:, 1:] = self.rawdat_df.iloc[:, 1:] / np.max(self.rawdat_df.iloc[:, 1:])

        # normlized by the maximum value of each row(sensor).
        if normalize == 2:
            for i in range(self.m):
                self.scale[i] = np.max(np.abs(self.rawdat[:, i]))
                self.dat[:, i] = self.rawdat[:, i] / np.max(np.abs(self.rawdat[:, i]))

    def _split(self, train, valid):
        # train and valid is the ratio of training set and validation set. test = 1 - train - valid

        self.train_idx = range(0, train)
        self.valid_idx = range(train, valid)
        self.test_idx = range(valid, self.n)

        train_set = range(self.hist_window + self.pred_window, train)
        valid_set = range(train, valid)
        test_set = range(valid, self.n)
        self.train = self._batchify(train_set)
        self.valid = self._batchify(valid_set)
        self.test = self._batchify(test_set)

    def _batchify(self, idx_set):
        n = len(idx_set)
        X = torch.zeros((n, self.hist_window, self.m))
        Y = torch.zeros((n, self.pred_window, self.m))

        for i in range(n):
            end = idx_set[i] - self.pred_window
            start = end - self.hist_window
            X[i, :, :] = torch.from_numpy(self.dat[start:end, :])
            Y[i, :, :] = torch.from_numpy(self.dat[end:idx_set[i], :])

        return [X, Y]

    def get_batches(self, inputs, targets, batch_size, shuffle=True):
        length = len(inputs)
        if shuffle:
            index = torch.randperm(length)
        else:
            index = torch.LongTensor(range(length))
        start_idx = 0
        while start_idx < length:
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            X = inputs[excerpt].to(self.device)
            Y = targets[excerpt].to(self.device)
            yield Variable(X), Variable(Y)
            start_idx += batch_size

