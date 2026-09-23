import os
import time

import numpy as np
import pandas as pd
import torch
from dtaidistance import dtw
from scipy.stats import pearsonr


class myInformer():
    def __init__(self, args, model, optim, criterion, evaluateL1, evaluateL2):
        self.args = args
        self.model = model
        self.optim = optim
        self.criterion = criterion
        self.evaluateL1 = evaluateL1
        self.evaluateL2 = evaluateL2
        self.y_true = None
        self.y_pred = None
        self._use_fast_dtw = getattr(dtw, "dtw_cc", None) is not None
        self.df_metrics = pd.DataFrame(index=range(self.args.epochs + 1), columns=["train_loss"])

    def train(self, Data):
        print('Training Informer ...')
        informer_sets = Data.get_informer_sets(label_len=self.args.label_len, freq=self.args.freq)
        best_val = float("inf")

        for epoch in range(1, self.args.epochs + 1):
            train_start_time = time.time()
            train_loss = self.train_(Data, informer_sets["train"])
            train_duration = time.time() - train_start_time

            valid_start_time = time.time()
            val_rmse_mean, val_mae_mean, val_corr_mean, val_dtw_mean, val_rmse_std, val_mae_std, val_corr_std, val_dtw_std = self.fast_evaluate_(Data, informer_sets["valid"])
            valid_duration = time.time() - valid_start_time

            print('\t| epoch: {:3d} | train_duration: {:5.2f}s | valid_duration: {:5.2f}s | train_loss {:5.6f} | valid rmse {:5.6f} | valid mae {:5.6f} | valid corr {:5.6f} | valid dtw {:5.6f}'.format(
                epoch, train_duration, valid_duration, train_loss, val_rmse_mean, val_mae_mean, val_corr_mean, val_dtw_mean))

            self.df_metrics.loc[epoch, ["train_loss"]] = train_loss
            self.df_metrics.loc[epoch, ["valid_rmse_mean", "val_mae_mean", "val_corr_mean", "val_dtw_mean"]] = val_rmse_mean, val_mae_mean, val_corr_mean, val_dtw_mean
            self.df_metrics.loc[epoch, ["valid_rmse_std", "val_mae_std", "val_corr_std", "val_dtw_std"]] = val_rmse_std, val_mae_std, val_corr_std, val_dtw_std

            if val_rmse_mean < best_val:
                torch.save(self.model.state_dict(), self.args.ck_path + "informer_model.pt")
                best_val = val_rmse_mean

            if epoch % 5 == 0:
                test_rmse_mean, test_mae_mean, test_corr_mean, test_dtw_mean, test_rmse_std, test_mae_std, test_corr_std, test_dtw_std = self.fast_evaluate_(Data, informer_sets["test"])
                print("\ttest rmse {:5.6f} | test mae {:5.6f} | test corr {:5.6f} | test dtw {:5.6f}".format(
                    test_rmse_mean, test_mae_mean, test_corr_mean, test_dtw_mean))
                self.df_metrics.loc[epoch, ["test_rmse_mean", "test_mae_mean", "test_corr_mean", "test_dtw_mean"]] = test_rmse_mean, test_mae_mean, test_corr_mean, test_dtw_mean
                self.df_metrics.loc[epoch, ["test_rmse_std", "test_mae_std", "test_corr_std", "test_dtw_std"]] = test_rmse_std, test_mae_std, test_corr_std, test_dtw_std
                self.df_metrics.to_csv(self.args.ck_path + "metrics.csv")

    def evaluate(self, Data):
        print('Evaluating Informer ...')
        model_path = os.path.join(self.args.ck_path, "informer_model.pt")
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=Data.device))
            print("\tloading informer_model.pt")

        informer_sets = Data.get_informer_sets(label_len=self.args.label_len, freq=self.args.freq)
        test_rmse_mean, test_mae_mean, test_corr_mean, test_dtw_mean, test_rmse_std, test_mae_std, test_corr_std, test_dtw_std = self.evaluate_(Data, informer_sets["test"])

        print("\ttest rmse {:5.6f} | test mae {:5.6f} | test corr {:5.6f} | test dtw {:5.6f}".format(
            test_rmse_mean, test_mae_mean, test_corr_mean, test_dtw_mean))
        self.df_metrics.loc[self.args.epochs + 1, ["test_rmse_mean", "test_mae_mean", "test_corr_mean", "test_dtw_mean"]] = test_rmse_mean, test_mae_mean, test_corr_mean, test_dtw_mean
        self.df_metrics.loc[self.args.epochs + 1, ["test_rmse_std", "test_mae_std", "test_corr_std", "test_dtw_std"]] = test_rmse_std, test_mae_std, test_corr_std, test_dtw_std
        self.df_metrics.to_csv(self.args.ck_path + "metrics.csv")

    def train_(self, data, informer_set):
        self.model.train()
        total_loss = 0
        n_batches = 0

        for X_enc, X_mark_enc, X_dec, X_mark_dec, Y in data.get_informer_batches(informer_set, self.args.batch_size, shuffle=True):
            self.optim.zero_grad()
            output = self.model(X_enc, X_mark_enc, X_dec, X_mark_dec)
            if isinstance(output, tuple):
                output = output[0]

            loss = self.criterion(output, Y)
            loss.backward()
            if self.args.clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
            self.optim.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def fast_evaluate_(self, data, informer_set):
        y_preds, y_trues = self._predict(data, informer_set)
        rmse_mean, rmse_std = self._stream_rmse(y_preds, y_trues)
        mae_mean, mae_std = self._stream_mae(y_preds, y_trues)
        corr_mean, corr_std = self._stream_corr(y_preds, y_trues)
        return rmse_mean, mae_mean, corr_mean, 0, rmse_std, mae_std, corr_std, 0

    def evaluate_(self, data, informer_set):
        y_preds, y_trues = self._predict(data, informer_set)
        rmse_mean, rmse_std, mae_mean, mae_std, corr_mean, corr_std, dtw_mean, dtw_std = self._full_metric_summary(y_preds, y_trues)
        self.y_pred, self.y_true = y_preds, y_trues
        return rmse_mean, mae_mean, corr_mean, dtw_mean, rmse_std, mae_std, corr_std, dtw_std

    def _predict(self, data, informer_set):
        self.model.eval()
        y_preds, y_trues = None, None

        with torch.no_grad():
            for X_enc, X_mark_enc, X_dec, X_mark_dec, Y in data.get_informer_batches(informer_set, self.args.batch_size, shuffle=False):
                output = self.model(X_enc, X_mark_enc, X_dec, X_mark_dec)
                if isinstance(output, tuple):
                    output = output[0]

                if y_preds is None:
                    y_preds = output.cpu()
                    y_trues = Y.cpu()
                else:
                    y_preds = torch.cat((y_preds, output.cpu()))
                    y_trues = torch.cat((y_trues, Y.cpu()))

        return y_preds.detach().numpy(), y_trues.detach().numpy()

    def _stream_rmse(self, y_preds, y_trues):
        stream_rmses = np.sqrt(((y_preds - y_trues) ** 2).mean(axis=(0, 1)))
        return np.mean(stream_rmses), np.std(stream_rmses)

    def _stream_mae(self, y_preds, y_trues):
        stream_maes = np.abs(y_preds - y_trues).mean(axis=(0, 1))
        return np.mean(stream_maes), np.std(stream_maes)

    def _stream_corr(self, y_preds, y_trues):
        y_preds_T = y_preds.transpose(1, 0, 2)
        y_trues_T = y_trues.transpose(1, 0, 2)
        sigma_p, sigma_g = y_preds_T.std(axis=0), y_trues_T.std(axis=0)
        mean_p, mean_g = y_preds_T.mean(axis=0), y_trues_T.mean(axis=0)
        index = (sigma_g != 0)
        batch_corrs = ((y_preds_T - mean_p) * (y_trues_T - mean_g)).mean(axis=0) / (sigma_p * sigma_g + 1e-16)
        corrs = batch_corrs[index]
        if len(corrs) == 0:
            return np.nan, np.nan
        return np.mean(corrs), np.std(corrs)

    def _full_metric_summary(self, y_preds, y_trues):
        rmses, maes, corrs, dtws = [], [], [], []

        for col in range(y_trues.shape[-1]):
            batch_rmses, batch_maes, batch_corrs, batch_dtws = [], [], [], []
            for b in range(y_trues.shape[0]):
                y_pred, y_true = y_preds[b, :, col], y_trues[b, :, col]
                if len(np.unique(y_true)) < 2:
                    continue

                batch_rmses.append(np.sqrt(((y_pred - y_true) ** 2).mean()))
                batch_maes.append(np.abs(y_pred - y_true).mean())
                batch_corrs.append(pearsonr(y_pred, y_true)[0])
                batch_dtws.append(self._dtw_distance(y_pred, y_true))

            if len(batch_rmses) < 2:
                continue

            rmses.append(np.mean(batch_rmses))
            maes.append(np.mean(batch_maes))
            corrs.append(np.mean(batch_corrs))
            dtws.append(np.mean(batch_dtws))

        return (
            np.mean(rmses), np.std(rmses),
            np.mean(maes), np.std(maes),
            np.mean(corrs), np.std(corrs),
            np.mean(dtws), np.std(dtws),
        )

    def _dtw_distance(self, y_pred, y_true):
        y_pred = y_pred.astype(np.double)
        y_true = y_true.astype(np.double)
        if self._use_fast_dtw:
            try:
                return dtw.distance_fast(y_pred, y_true)
            except Exception:
                self._use_fast_dtw = False
        return dtw.distance(y_pred, y_true)
