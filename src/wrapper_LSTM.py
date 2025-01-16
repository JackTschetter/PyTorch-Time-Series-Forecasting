import os
import torch
import time
import math
import tqdm
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from dtaidistance import dtw
from multiprocessing import Pool


class myLSTM():
    def __init__(self, args, model, optim, criterion, evaluateL1, evaluateL2):
        self.args = args
        self.model = model
        self.optim = optim
        self.criterion = criterion
        self.evaluateL1 = evaluateL1
        self.evaluateL2 = evaluateL2
        self.y_true = None
        self.y_pred = None
        self.df_metrics = pd.DataFrame(index=range(self.args.epochs + 1), columns=["train_loss"])

    def train(self, Data):
        print('Training ...')
        best_val, best_corr = 10000000, -10000000
        for epoch in range(1, self.args.epochs + 1):
            # Train
            train_start_time = time.time()
            train_loss = self.train_(Data, Data.train[0], Data.train[1])
            train_duration = time.time() - train_start_time

            # Validation
            valid_start_time = time.time()
            val_rmse_mean, val_mae_mean, val_corr_mean, val_dtw_mean, val_rmse_std, val_mae_std, val_corr_std, val_dtw_std = self.fast_evaluate_(Data, Data.valid[0], Data.valid[1])
            valid_duration = time.time() - valid_start_time
            print('\t| epoch: {:3d} | train_duration: {:5.2f}s | valid_duration: {:5.2f}s | train_loss {:5.6f} | valid rmse {:5.6f} | valid mae {:5.6f} | valid corr  {:5.6f} | valid dtw  {:5.6f}'.format(
                epoch, train_duration, valid_duration, train_loss, val_rmse_mean, val_mae_mean, val_corr_mean, val_dtw_mean))
            self.df_metrics.loc[epoch, ["train_loss"]] = train_loss
            self.df_metrics.loc[epoch, ["valid_rmse_mean", "val_mae_mean", "val_corr_mean", "val_dtw_mean"]] = val_rmse_mean, val_mae_mean, val_corr_mean, val_dtw_mean
            self.df_metrics.loc[epoch, ["valid_rmse_std", "val_mae_std", "val_corr_std", "val_dtw_std"]] = val_rmse_std, val_mae_std, val_corr_std, val_dtw_std

            # if val_rse < best_val and epoch >= 50:
            #     torch.save(self.model, self.args.ck_path + "model")
            #     best_val = val_rse
            if val_corr_mean > best_corr and epoch >= 50:
                torch.save(self.model, self.args.ck_path + "model")
                best_corr = val_corr_mean

            # Evaluation
            if epoch % 5 == 0:
                test_start_time = time.time()
                test_rmse_mean, test_mae_mean, test_corr_mean, test_dtw_mean, test_rmse_std, test_mae_std, test_corr_std, test_dtw_std = self.fast_evaluate_(Data, Data.test[0], Data.test[1])
                test_duration = time.time() - test_start_time
                print("\ttest duration {:5.2f}s |test rmse {:5.6f} | test mae {:5.6f} | test corr {:5.6f} | test dtw {:5.6f}".format(
                    test_duration, test_rmse_mean, test_mae_mean, test_corr_mean, test_dtw_mean))
                self.df_metrics.loc[epoch, ["test_rmse_mean", "test_mae_mean", "test_corr_mean", "test_dtw_mean"]] = test_rmse_mean, test_mae_mean, test_corr_mean, test_dtw_mean
                self.df_metrics.loc[epoch, ["test_rmse_std", "test_mae_std", "test_corr_std", "test_dtw_std"]] = test_rmse_std, test_mae_std, test_corr_std, test_dtw_std
                self.df_metrics.to_csv(self.args.ck_path + "metrics.csv")

    def evaluate(self, Data):
        print('Evaluating ...')
        # Load the best saved model.
        for filename in os.listdir(self.args.ck_path):
            if "model" in filename:
                with open(self.args.ck_path+filename, 'rb') as f:
                    self.model = torch.load(f)
                    print("\tloading "+filename)

        # Evaluate the model
        test_rmse_mean, test_mae_mean, test_corr_mean, test_dtw_mean, test_rmse_std, test_mae_std, test_corr_std, test_dtw_std = self.evaluate_(Data, Data.test[0], Data.test[1])

        # Save results
        print("\ttest rmse {:5.6f} | test mae {:5.6f} | test corr {:5.6f} | test dtw {:5.6f}".format(
            test_rmse_mean, test_mae_mean, test_corr_mean, test_dtw_mean))
        self.df_metrics.loc[self.args.epochs+1, ["test_rmse_mean", "test_mae_mean", "test_corr_mean", "test_dtw_mean"]] = test_rmse_mean, test_mae_mean, test_corr_mean, test_dtw_mean
        self.df_metrics.loc[self.args.epochs+1, ["test_rmse_std", "test_mae_std", "test_corr_std", "test_dtw_std"]] = test_rmse_std, test_mae_std, test_corr_std, test_dtw_std
        self.df_metrics.to_csv(self.args.ck_path + "metrics.csv")

    ##########################################################
    # Support Functions
    ##########################################################
    def train_(self, data, X, Y):
        model, criterion, optim, batch_size = self.model, self.criterion, self.optim, self.args.batch_size

        model.train()
        total_loss = 0
        n_batches = 0
        for X, Y in data.get_batches(X, Y, batch_size, True):
            model.zero_grad()
            output = model(X)
            loss = criterion(output, Y)
            loss.backward()
            optim.step()
            total_loss += loss.item()
            n_batches += 1
        return total_loss / n_batches

    def fast_evaluate_(self, data, X, Y):
        # This is the fast (matrix-based) implementation of the evaluation function
        # The RMSE, MAE, CORR is same as for-loop-based version

        def get_accumulate_rmse(preds, tests):
            # Input size: (batch size, pred_length, streams); Output size: (streams)
            sample_rmses = torch.sqrt(torch.mean(evaluateL2(preds, tests), dim=1))  # size: (batch size, streams)
            batch_rmses = torch.sum(sample_rmses, dim=0)                            # size: (streams)
            return batch_rmses

        def get_accumulate_mae(preds, tests):
            # Input size: (batch size, pred_length, streams); Output size: (streams)
            sample_maes = torch.mean(evaluateL1(preds, tests), dim=1)  # size: (batch size, streams)
            batch_maes = torch.sum(sample_maes, dim=0)                 # size: (streams)
            return batch_maes

        def get_accumulate_corr(preds, tests):
            # Input size: (batch size, pred_length, streams); Output size: (streams)
            y_preds_T = torch.transpose(preds, 0, 1)
            y_tests_T = torch.transpose(tests, 0, 1)
            sigma_p, sigma_g = y_preds_T.std(dim=0, unbiased=False), y_tests_T.std(dim=0, unbiased=False)  # std functions in pytorch and numpy are different
            mean_p, mean_g = y_preds_T.mean(dim=0), y_tests_T.mean(dim=0)
            index = (sigma_g != 0)
            sample_corrs = ((y_preds_T-mean_p)*(y_tests_T-mean_g)).mean(dim=0)/(sigma_p*sigma_g+1e-16)  # size: (batch_size, streams)
            corrs = torch.sum(sample_corrs[index], dim=0)        # size: (1)
            valid_samples = torch.sum(index)                     # size: (1)
            return corrs, valid_samples

        model, evaluateL2, evaluateL1, batch_size = self.model, self.evaluateL2, self.evaluateL1, self.args.batch_size
        model.eval()
        samples, valid_samples = len(Y), None
        batch_rmses, batch_maes, corrs = None, None, None

        # Accumulate all batches
        for batch_X, batch_Y in data.get_batches(X, Y, batch_size=batch_size, shuffle=False):
            output = model(batch_X)

            # Stack the outputs
            if batch_rmses is None:
                batch_rmses = get_accumulate_rmse(output, batch_Y).detach()
                batch_maes = get_accumulate_mae(output, batch_Y).detach()
                r1, r2 = get_accumulate_corr(output, batch_Y)
                corrs = r1.detach()
                valid_samples = r2.detach()
            else:
                batch_rmses += get_accumulate_rmse(output, batch_Y).detach()
                batch_maes += get_accumulate_mae(output, batch_Y).detach()
                r1, r2 = get_accumulate_corr(output, batch_Y)
                corrs += r1.detach()
                valid_samples += r2.detach()

        # Calculate the metrics
        batch_rmses = batch_rmses / samples
        batch_maes = batch_maes / samples
        corrs = corrs / valid_samples
        rmse_mean, rmse_std = torch.mean(batch_rmses).item(), torch.std(batch_rmses).item()
        mae_mean, mae_std = torch.mean(batch_maes).item(), torch.std(batch_maes).item()
        corr_mean, corr_std = corrs.item(), 0

        dtw_mean, dtw_std = 0, 0
        return rmse_mean, mae_mean, corr_mean, dtw_mean, rmse_std, mae_std, corr_std, dtw_std

    def evaluate_(self, data, X, Y, multiprocessing=True):
        # This is the fast (for-loop-based) implementation of the evaluation function.
        # Besides RMSE, MAE, CORR, we further calculate DTW

        model, evaluateL2, evaluateL1, batch_size = self.model, self.evaluateL2, self.evaluateL1, self.args.batch_size
        model.eval()
        y_preds, y_trues = None, None

        # Accumulate all batches
        for X, Y in data.get_batches(X, Y, batch_size=batch_size, shuffle=False):
            output = model(X)
            if y_preds is None:
                y_preds = output.cpu()
                y_trues = Y.cpu()
            else:
                y_preds = torch.cat((y_preds, output.cpu()))
                y_trues = torch.cat((y_trues, Y.cpu()))
        y_trues = y_trues.detach().numpy()
        y_preds = y_preds.detach().numpy()

        # Loop the data streams (features)
        rmses, maes, corrs, btws = [], [], [], []
        if multiprocessing:
            inputs = []
            for col in range(y_trues.shape[-1]):
                inputs.append((y_preds[:, :, col], y_trues[:, :, col]))

            p = Pool(48)
            p_outs = list(tqdm.tqdm(p.imap(mp_calculate_metrics, inputs), total=y_trues.shape[-1]))
            p.close()
            p.join()

            for p_out in p_outs:
                batch_rmses, batch_maes, batch_corrs, batch_btws = p_out
                if len(batch_rmses) < 2:
                    # Sometimes we will skip too much cutoffs, leading to empty list
                    continue
                rmses.append(np.mean(batch_rmses))
                maes.append(np.mean(batch_maes))
                corrs.append(np.mean(batch_corrs))
                btws.append(np.mean(batch_btws))

        else:
            for col in tqdm.tqdm(range(y_trues.shape[-1]), disable=False):
                # Loop the batches
                batch_rmses, batch_maes, batch_corrs, batch_btws = [], [], [], []
                for b in range(y_trues.shape[0]):
                    y_pred, y_true = y_preds[b, :, col], y_trues[b, :, col]
                    if len(np.unique(y_true)) < 2:
                        # the same y_ture (e.g., all the empty value) cannot calculate the pearsonr
                        continue
                    batch_rmses.append(np.sqrt(((y_pred - y_true) ** 2).mean()))
                    batch_maes.append(np.abs(y_pred - y_true).mean())
                    batch_corrs.append(pearsonr(y_pred, y_true)[0])
                    batch_btws.append(dtw.distance_fast(y_pred.astype(np.double), y_true.astype(np.double)))

                if len(batch_rmses) < 2:
                    # Sometimes we will skip too much cutoffs, leading to empty list
                    continue
                rmses.append(np.mean(batch_rmses))
                maes.append(np.mean(batch_maes))
                corrs.append(np.mean(batch_corrs))
                btws.append(np.mean(batch_btws))

        # Calculate the mean and std over each stream and batch
        rmse_mean, rmse_std = np.mean(rmses), np.std(rmses)
        mae_mean, mae_std = np.mean(maes), np.std(maes)
        corr_mean, corr_std = np.mean(corrs), np.std(corrs)
        dtw_mean, dtw_std = np.mean(btws), np.std(btws)

        self.y_pred, self.y_true = y_preds, y_trues
        return rmse_mean, mae_mean, corr_mean, dtw_mean, rmse_std, mae_std, corr_std, dtw_std


def mp_calculate_metrics(inputs):
    y_preds_stream, y_trues_stream = inputs

    # Loop the batches
    batch_rmses, batch_maes, batch_corrs, batch_btws = [], [], [], []
    for b in range(y_trues_stream.shape[0]):
        y_pred, y_true = y_preds_stream[b, :], y_trues_stream[b, :]

        if len(np.unique(y_true)) < 2:
            # the same y_ture (e.g., all the empty value) cannot calculate the pearsonr
            continue

        batch_rmses.append(np.sqrt(((y_pred - y_true) ** 2).mean()))
        batch_maes.append(np.abs(y_pred - y_true).mean())
        batch_corrs.append(pearsonr(y_pred, y_true)[0])
        batch_btws.append(dtw.distance_fast(y_pred.astype(np.double), y_true.astype(np.double)))

    return batch_rmses, batch_maes, batch_corrs, batch_btws