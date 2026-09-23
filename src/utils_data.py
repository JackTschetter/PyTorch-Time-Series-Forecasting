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
        self.scale = torch.ones(self.m).to(device)
        self._time_feature_cache = {}
        self._normalized(normalize)
        self._split(int(train * self.n), int((train + valid) * self.n))

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

        self.train_set = range(self.hist_window + self.pred_window, train)
        self.valid_set = range(train, valid)
        self.test_set = range(valid, self.n)
        self.train = self._batchify(self.train_set)
        self.valid = self._batchify(self.valid_set)
        self.test = self._batchify(self.test_set)

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

    def get_informer_sets(self, label_len, freq='h'):
        if label_len <= 0:
            raise ValueError("label_len must be positive for Informer.")
        if label_len > self.hist_window:
            raise ValueError("label_len must be less than or equal to hist_window for Informer.")

        return {
            "train": self._batchify_informer(self.train_set, label_len, freq),
            "valid": self._batchify_informer(self.valid_set, label_len, freq),
            "test": self._batchify_informer(self.test_set, label_len, freq),
        }

    def _batchify_informer(self, idx_set, label_len, freq):
        n = len(idx_set)
        X_enc = torch.zeros((n, self.hist_window, self.m))
        X_mark_enc = torch.zeros((n, self.hist_window, self._time_feature_dim(freq)))
        X_dec = torch.zeros((n, label_len + self.pred_window, self.m))
        X_mark_dec = torch.zeros((n, label_len + self.pred_window, self._time_feature_dim(freq)))
        Y = torch.zeros((n, self.pred_window, self.m))

        time_features = self._get_time_features(freq)
        for i in range(n):
            pred_end = idx_set[i]
            pred_start = pred_end - self.pred_window
            hist_start = pred_start - self.hist_window
            label_start = pred_start - label_len

            X_enc[i, :, :] = torch.from_numpy(self.dat[hist_start:pred_start, :])
            X_mark_enc[i, :, :] = torch.from_numpy(time_features[hist_start:pred_start, :])
            X_dec[i, :label_len, :] = torch.from_numpy(self.dat[label_start:pred_start, :])
            X_mark_dec[i, :, :] = torch.from_numpy(time_features[label_start:pred_end, :])
            Y[i, :, :] = torch.from_numpy(self.dat[pred_start:pred_end, :])

        return [X_enc, X_mark_enc, X_dec, X_mark_dec, Y]

    def get_informer_batches(self, informer_set, batch_size, shuffle=True):
        length = len(informer_set[0])
        if shuffle:
            index = torch.randperm(length)
        else:
            index = torch.LongTensor(range(length))
        start_idx = 0
        while start_idx < length:
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            yield [tensor[excerpt].to(self.device) for tensor in informer_set]
            start_idx += batch_size

    def _get_time_features(self, freq):
        freq = freq.lower()
        if freq not in self._time_feature_cache:
            dates = pd.DatetimeIndex(pd.to_datetime(self.rawdat_df["ds"]))
            if freq != 'h':
                raise ValueError("This Informer integration currently supports hourly frequency only: freq='h'.")

            features = np.stack(
                [
                    dates.month,
                    dates.day,
                    dates.dayofweek,
                    dates.hour,
                ],
                axis=1,
            ).astype(np.float32)
            self._time_feature_cache[freq] = features
        return self._time_feature_cache[freq]

    def _time_feature_dim(self, freq):
        if freq.lower() != 'h':
            raise ValueError("This Informer integration currently supports hourly frequency only: freq='h'.")
        return 4

