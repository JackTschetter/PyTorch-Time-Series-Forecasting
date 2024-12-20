import torch
import time
import math
import os
import numpy as np
import pandas as pd


# wrap up
class myUnivarDIVIDE():
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
            self.epoch = epoch

            # Train
            train_start_time = time.time()
            train_loss = self.train_(Data, Data.train[0], Data.train[1])
            train_duration_time = time.time() - train_start_time

            # Validation
            val_rse, val_rae, val_corr = self.evaluate_(Data, Data.valid[0], Data.valid[1], )
            print('\t| epoch: {:3d} | train_duration: {:5.2f}s | train_loss {:5.4f} | valid rse {:5.4f} | valid rae {:5.4f} | valid corr  {:5.4f} |'.format(
                    epoch, train_duration_time, train_loss, val_rse, val_rae, val_corr))
            self.df_metrics.loc[epoch, ["train_loss", "valid_rse", "val_rae", "val_corr"]] = train_loss, val_rse, val_rae, val_corr

            # if val_rse < best_val and epoch >= 50:
            #     for id, model in enumerate(self.models):
            #         torch.save(model, self.args.ck_path + "model_{}".format(str(id)))
            #     best_val = val_rse
            if val_corr > best_corr and epoch >= 50:
                for id, model in enumerate(self.models):
                    torch.save(model, self.args.ck_path + "model_{}".format(str(id)))
                best_corr = val_corr

            # Evaluation
            if epoch % 5 == 0:
                test_rse, test_rae, test_corr = self.evaluate_(Data, Data.test[0], Data.test[1])
                print("\ttest rse {:5.4f} | test rae {:5.4f} | test corr {:5.4f}".format(test_rse, test_rae, test_corr))
                self.df_metrics.loc[epoch, ["test_rse", "test_rae", "test_corr"]] = test_rse, test_rae, test_corr

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
        test_acc, test_rae, test_corr = self.evaluate_(Data, Data.test[0], Data.test[1])

        # Save results
        print("\ttest rse {:5.4f} | test rae {:5.4f} | test corr {:5.4f}".format(test_acc, test_rae, test_corr))
        self.df_metrics.to_csv(self.args.ck_path + "metrics.csv")

    ##########################################################
    # Support Functions
    ##########################################################
    def train_(self, data, X, Y):
        local_models, criterion, local_optims, batch_size = self.models, self.criterion, self.optims, self.args.batch_size

        for model in local_models:
            model.train()
        total_loss = 0
        n_batches = 0

        for X, Y in data.get_batches(X, Y, batch_size, True):
            # Update local models
            local_outs = []
            chunks = np.array_split(np.arange(X.size()[-1]), len(local_models))
            for k in range(len(local_models)):
                # Retrieve k-th node information
                local_model_k = local_models[k]
                Xk, Yk = X[:, :, chunks[k]], Y[:, :, chunks[k]]

                # Feed forward the k-th local node and save it for global model
                local_model_k.zero_grad()
                local_out_k = local_model_k(Xk)  # size: (#samples, pred hours, local data dim)
                local_outs.append(local_out_k)
            # outs = torch.stack(local_outs).permute(1, 2, 0)
            outs = torch.cat(local_outs, dim=-1)

            # Back propogation
            loss = criterion(outs, Y)
            loss.backward()

            # Update local models
            for k in range(len(local_models)):
                local_optim_k = local_optims[k]
                local_optim_k.step()

            # Logging
            total_loss += loss.item()
            n_batches += 1
        return total_loss / n_batches

    def evaluate_(self, data, X, Y):
        local_models, evaluateL2, evaluateL1, batch_size = self.models, self.evaluateL2, self.evaluateL1, self.args.batch_size

        for model in local_models:
            model.eval()
        global_evaluateL2 = 0
        global_evaluateL1 = 0
        n_batches = 0
        predict = None
        test = None

        for X, Y in data.get_batches(X, Y, batch_size, False):
            # Local models
            local_outs = []
            chunks = np.array_split(np.arange(X.size()[-1]), len(local_models))
            for k in range(len(local_models)):
                # retrieve k-th node information
                local_model_k = local_models[k]
                Xk, Yk = X[:, :, chunks[k]], Y[:, :, chunks[k]]

                # Feed forward the k-th local node and save it for global model
                local_output_k = local_model_k(Xk)
                local_outs.append(local_output_k)
            outs = torch.cat(local_outs, dim=-1)

            # Evaluate the results
            global_evaluateL1 += evaluateL1(outs, Y).item()
            global_evaluateL2 += evaluateL2(outs, Y).item()
            n_batches += 1

            # Accumulate all batches
            if predict is None:
                predict = outs.cpu()
                test = Y.cpu()
            else:
                predict = torch.cat((predict, outs.cpu()))
                test = torch.cat((test, Y.cpu()))

        rse = math.sqrt(global_evaluateL2 / n_batches)
        rae = global_evaluateL1 / n_batches

        predict = predict.data.numpy()
        Ytest = test.data.numpy()
        sigma_p = predict.std(axis=0)
        sigma_g = Ytest.std(axis=0)
        mean_p = predict.mean(axis=0)
        mean_g = Ytest.mean(axis=0)
        index = (sigma_g != 0)
        correlation = ((predict - mean_p) * (Ytest - mean_g)).mean(axis=0) / (sigma_p * sigma_g + 1e-16)
        correlation = (correlation[index]).mean()

        self.y_pred = predict
        self.y_true = Ytest
        return rse, rae, correlation


