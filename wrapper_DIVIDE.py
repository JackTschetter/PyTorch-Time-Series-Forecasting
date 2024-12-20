import torch
import time
import math
import os
import torch.nn as nn
import numpy as np
import pandas as pd
import tqdm
from scipy.stats import pearsonr
from dtaidistance import dtw


# MLP
class GE_MLP(nn.Module):
    def __init__(self, args):
        super(GE_MLP, self).__init__()
        self.pred_window = args.pred_window
        self.n_local = args.n_local
        self.data_dim = args.data_dim
        self.local_emb_dim = args.local_emb_dim
        self.input_dim = args.local_emb_dim * args.n_local
        self.output_dim = args.data_dim * args.pred_window

        self.model = nn.Sequential(nn.Linear(self.input_dim, 128),
                                   nn.LayerNorm(128),
                                   nn.ReLU(),
                                   nn.Linear(128, self.output_dim))

    def forward(self, x):
        batch_size = x.size(0)

        x = x.view(batch_size, -1)
        out = self.model(x)
        out = out.view(batch_size, self.pred_window, self.data_dim)
        return out


# wrap up
class myDIVIDE():
    def __init__(self, args, models, optims, criterion, evaluateL1, evaluateL2):
        self.args = args
        self.models = models
        self.optims = optims
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
            self.cur_epoch = epoch

            # Train
            train_start_time = time.time()
            train_loss = self.train_(Data, Data.train[0], Data.train[1])
            train_duration = time.time() - train_start_time

            # Validation
            valid_start_time = time.time()
            val_rmse_mean, val_mae_mean, val_corr_mean, val_dtw_mean, val_rmse_std, val_mae_std, val_corr_std, val_dtw_std = self.fast_evaluate_(Data, Data.valid[0], Data.valid[1])
            valid_duration = time.time() - valid_start_time
            print('\t| epoch: {:3d} | train_duration: {:5.2f}s | valid_duration: {:5.2f}s| train_loss {:5.6f} | valid rmse {:5.6f} | valid mae {:5.6f} | valid corr  {:5.6f} | valid dtw  {:5.6f}'.format(
                epoch, train_duration, valid_duration, train_loss, val_rmse_mean, val_mae_mean, val_corr_mean, val_dtw_mean))
            self.df_metrics.loc[epoch, ["train_loss"]] = train_loss
            self.df_metrics.loc[epoch, ["valid_rmse_mean", "val_mae_mean", "val_corr_mean", "val_dtw_mean"]] = val_rmse_mean, val_mae_mean, val_corr_mean, val_dtw_mean
            self.df_metrics.loc[epoch, ["valid_rmse_std", "val_mae_std", "val_corr_std", "val_dtw_std"]] = val_rmse_std, val_mae_std, val_corr_std, val_dtw_std

            # if val_rse < best_val and epoch >= 50:
            #     for id, model in enumerate(self.models):
            #         torch.save(model, self.args.ck_path + "model_{}".format(str(id)))
            #     best_val = val_rse
            if val_corr_mean > best_corr and epoch >= 50:
                for id, model in enumerate(self.models):
                    torch.save(model, self.args.ck_path + "model_{}".format(str(id)))
                best_corr = val_corr_mean

            # Evaluation
            if epoch % 5 == 0:
                test_rmse_mean, test_mae_mean, test_corr_mean, test_dtw_mean, test_rmse_std, test_mae_std, test_corr_std, test_dtw_std = self.fast_evaluate_(Data, Data.test[0], Data.test[1])
                print("\ttest rmse {:5.6f} | test mae {:5.6f} | test corr {:5.6f} | test dtw {:5.6f}".format(
                    test_rmse_mean, test_mae_mean, test_corr_mean, test_dtw_mean))
                self.df_metrics.loc[epoch, ["test_rmse_mean", "test_mae_mean", "test_corr_mean", "test_dtw_mean"]] = test_rmse_mean, test_mae_mean, test_corr_mean, test_dtw_mean
                self.df_metrics.loc[epoch, ["test_rmse_std", "test_mae_std", "test_corr_std", "test_dtw_std"]] = test_rmse_std, test_mae_std, test_corr_std, test_dtw_std
                self.df_metrics.to_csv(self.args.ck_path + "metrics.csv")

    def evaluate(self, Data):
        print('Evaluating ...')
        # Load the best saved model.
        i = 0
        for filename in sorted(os.listdir(self.args.ck_path)):
            if "model" in filename:
                with open(self.args.ck_path + filename, 'rb') as f:
                    self.models[i] = torch.load(f)
                    print("\tloading "+filename)
                i += 1

        # Eval the model
        test_rmse_mean, test_mae_mean, test_corr_mean, test_dtw_mean, test_rmse_std, test_mae_std, test_corr_std, test_dtw_std = self.evaluate_(Data, Data.test[0], Data.test[1])

        # Save results
        print("\ttest rmse {:5.6f} | test mae {:5.6f} | test corr {:5.6f} | test dtw {:5.6f}".format(
            test_rmse_mean, test_mae_mean, test_corr_mean, test_dtw_mean))
        self.df_metrics.loc[self.args.epochs + 1, ["test_rmse_mean", "test_mae_mean", "test_corr_mean", "test_dtw_mean"]] = test_rmse_mean, test_mae_mean, test_corr_mean, test_dtw_mean
        self.df_metrics.loc[self.args.epochs + 1, ["test_rmse_std", "test_mae_std", "test_corr_std", "test_dtw_std"]] = test_rmse_std, test_mae_std, test_corr_std, test_dtw_std
        self.df_metrics.to_csv(self.args.ck_path + "metrics.csv")


    ##########################################################
    # Support Functions
    ##########################################################
    def train_(self, data, X, Y):
        models, criterion, optims, batch_size = self.models, self.criterion, self.optims, self.args.batch_size
        for model in models:
            model.train()
        global_model, local_models = models[0], models[1:]
        global_optim, local_optims = optims[0], optims[1:]
        total_global_loss = 0
        n_batches = 0

        for X, Y in data.get_batches(X, Y, batch_size, True):
            # Update local models
            local_embs = []
            chunks = np.array_split(np.arange(X.size()[-1]), len(local_models))
            for k in range(len(local_models)):
                # Retrieve k-th node information
                local_model_k = local_models[k]
                Xk, Yk = X[:, :, chunks[k]], Y[:, :, chunks[k]]

                # Feed forward the k-th local node and save it for global model
                local_model_k.zero_grad()
                local_output_k = local_model_k(Xk)
                local_embs.append(local_output_k)
            local_embs = torch.cat(local_embs, dim=-1)

            # Feed forward global model
            global_model.zero_grad()
            global_output = global_model(local_embs)

            # Back propogation
            global_loss = criterion(global_output, Y)
            global_loss.backward()

            # Update local and global model
            for k in range(len(local_models)):
                # if self.cur_epoch < 50 and k in [0, 1, 2, 3]:
                # Local model cold start
                #     continue
                local_optim_k = local_optims[k]
                local_optim_k.step()
            global_optim.step()

            # Logging
            total_global_loss += global_loss.item()
            n_batches += 1
        return total_global_loss / n_batches

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

        models, evaluateL2, evaluateL1, batch_size = self.models, self.evaluateL2, self.evaluateL1, self.args.batch_size
        for model in models:
            model.eval()
        global_model, local_models = models[0], models[1:]
        samples = 0
        batch_rmses, batch_maes = None, None
        y_preds, y_tests = None, None

        # Accumulate all batches
        for X, Y in data.get_batches(X, Y, batch_size=batch_size, shuffle=False):
            samples += len(X)

            # Local models
            local_embs = []
            chunks = np.array_split(np.arange(X.size()[-1]), len(local_models))
            for k in range(len(local_models)):
                # retrieve k-th node information
                local_model_k = local_models[k]
                Xk, Yk = X[:, :, chunks[k]], Y[:, :, chunks[k]]

                # Feed forward the k-th local node and save it for global model
                local_output_k = local_model_k(Xk)
                local_embs.append(local_output_k)
            local_embs = torch.cat(local_embs, dim=-1)

            # Global model
            global_output = global_model(local_embs)

            # Stack the outputs
            if y_preds is None:
                batch_rmses = get_accumulate_rmse(global_output, Y)
                batch_maes = get_accumulate_mae(global_output, Y)
                y_preds = global_output.cpu()
                y_tests = Y.cpu()
            else:
                batch_rmses += get_accumulate_rmse(global_output, Y)
                batch_maes += get_accumulate_mae(global_output, Y)
                y_preds = torch.cat((y_preds, global_output.cpu()))
                y_tests = torch.cat((y_tests, Y.cpu()))

        y_preds = y_preds.data.numpy()
        y_tests = y_tests.data.numpy()
        self.y_pred, self.y_true = y_preds, y_tests

        # Calculate the metrics
        batch_rmses = batch_rmses / samples
        batch_maes = batch_maes / samples
        rmse_mean, rmse_std = torch.mean(batch_rmses).item(), torch.std(batch_rmses).item()
        mae_mean, mae_std = torch.mean(batch_maes).item(), torch.std(batch_maes).item()

        y_preds_T = y_preds.transpose(1, 0, 2)
        y_tests_T = y_tests.transpose(1, 0, 2)
        sigma_p, sigma_g = y_preds_T.std(axis=0), y_tests_T.std(axis=0)
        mean_p, mean_g = y_preds_T.mean(axis=0), y_tests_T.mean(axis=0)
        index = (sigma_g != 0)
        batch_corrs = ((y_preds_T - mean_p) * (y_tests_T - mean_g)).mean(axis=0) / (sigma_p * sigma_g + 1e-16)  # size: (total samples, streams)
        corrs = np.mean(batch_corrs[index], axis=0)
        corr_mean, corr_std = corrs.mean(), corrs.std()

        dtw_mean, dtw_std = 0, 0
        return rmse_mean, mae_mean, corr_mean, dtw_mean, rmse_std, mae_std, corr_std, dtw_std

    def evaluate_(self, data, X, Y):
        # This is the fast (for-loop-based) implementation of the evaluation function.
        # Besides RMSE, MAE, CORR, we further calculate DTW

        models, evaluateL2, evaluateL1, batch_size = self.models, self.evaluateL2, self.evaluateL1, self.args.batch_size
        for model in models:
            model.eval()
        global_model, local_models = models[0], models[1:]
        y_preds, y_trues = None, None

        # Accumulate all batches
        for X, Y in data.get_batches(X, Y, batch_size=batch_size, shuffle=False):
            # Local models
            local_embs = []
            chunks = np.array_split(np.arange(X.size()[-1]), len(local_models))
            for k in range(len(local_models)):
                # retrieve k-th node information
                local_model_k = local_models[k]
                Xk, Yk = X[:, :, chunks[k]], Y[:, :, chunks[k]]

                # Feed forward the k-th local node and save it for global model
                local_output_k = local_model_k(Xk)
                local_embs.append(local_output_k)
            local_embs = torch.cat(local_embs, dim=-1)

            # Global model
            global_output = global_model(local_embs)

            # Stack the outputs
            if y_preds is None:
                y_preds = global_output.cpu()
                y_trues = Y.cpu()
            else:
                y_preds = torch.cat((y_preds, global_output.cpu()))
                y_trues = torch.cat((y_trues, Y.cpu()))
        y_trues = y_trues.detach().numpy()
        y_preds = y_preds.detach().numpy()

        # Loop the data streams (features)
        rmses, maes, corrs, btws = [], [], [], []
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


