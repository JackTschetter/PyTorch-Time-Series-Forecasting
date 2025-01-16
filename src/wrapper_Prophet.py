
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from dtaidistance import dtw
from prophet.diagnostics import performance_metrics
import tqdm
import time
import pickle


# wrap up
class myProphet():
    def __init__(self, args, model, FLAG_CROSS_VALID):
        self.args = args
        self.model = model
        self.y_true = None
        self.y_pred = None
        self.df_metrics = pd.DataFrame(index=range(1), columns=["test_rmse_mean"])
        self.FLAG_CROSS_VALID = FLAG_CROSS_VALID
        self.FLAG_NEW_TRAIN = 0

    def train(self, Data):
        start_time = time.time()

        print('Training ...')
        if self.FLAG_CROSS_VALID:
            df_train = Data.dat_df
            # df_train = Data.dat_df.iloc[Data.train_idx]
            # self.model.fit(df_train)
            self.model.mp_fit(df_train)

        else:
            df_train = Data.dat_df.iloc[Data.train_idx]
            self.model.fit(df_train)

            df_valid = Data.dat_df.iloc[Data.valid_idx].head(self.args.pred_window)
            valid_y_pred_dict = self.model.predict(df_valid)
            val_rmse, val_mae, val_corr, val_dtw = self.evaluate_(valid_y_pred_dict, df_valid)
            print("\nvalid rmse {:5.6f} | valid mae {:5.6f} | valid corr {:5.6f} | valid dtw {:5.6f}".format(val_rmse, val_mae, val_corr, val_dtw))

        self.FLAG_NEW_TRAIN = 1
        print("\tTotal training duration: {:.2f}s".format(time.time()-start_time))

    def evaluate(self, Data):
        # Save and Load the model
        if self.FLAG_NEW_TRAIN == 0:
            print("Loading from the disk...")
            for column, _ in self.model.model_pool.items():
                with open(self.args.model_path + 'prophet_{}.pckl'.format(column), 'rb') as fin:
                    self.model.model_pool[column] = pickle.load(fin)

        else:
            print("Saving the new trained model to the disk...")
            for column, model_i in self.model.model_pool.items():
                with open(self.args.ck_path + 'prophet_{}.pckl'.format(column), 'wb') as fout:
                    pickle.dump(model_i, fout)

        # Evaluation
        start_time = time.time()
        if self.FLAG_CROSS_VALID:
            print('Evaluating via cross validation ...')
            init = "{} hours".format(Data.valid_idx[-1])
            peri = "{} hours".format(max(336, self.args.pred_window))
            hori = "{} hours".format(self.args.pred_window)
            cross_valid_dict = self.model.cross_validation(initial=init, period=peri, horizon=hori, parallel="processes")
            print('Calculating metrics ...')
            test_rmse, test_mae, test_corr, test_dtw = self.CV_evaluate_(cross_valid_dict)
            print("test rmse {:5.6f} | test mae {:5.6f} | test corr {:5.6f} | test dtw {:5.6f}".format(test_rmse, test_mae, test_corr, test_dtw))

        else:
            print('Evaluating ...')
            df_test = Data.dat_df.iloc[Data.test_idx]
            # df_test = Data.dat_df.iloc[Data.test_idx].head(self.args.pred_window)
            test_y_pred_dict = self.model.predict(df_test)
            print('Calculating metrics ...')
            test_rmse, test_mae, test_corr, test_dtw = self.evaluate_(test_y_pred_dict, df_test)
            print("test rmse {:5.6f} | test mae {:5.6f} | test corr {:5.6f} | test dtw {:5.6f}".format(test_rmse, test_mae, test_corr, test_dtw))
        print("\tTotal evaluation duration: {:.2f}s".format(time.time()-start_time))


    ##########################################################
    # Support Functions
    ##########################################################
    def evaluate_(self, valid_y_pred_dict, df_true):

        # Loop the data streams (features)
        rmses, maes, corrs, dtws = [], [], [], []
        for col_name in tqdm.tqdm(valid_y_pred_dict.keys()):

            df_i = valid_y_pred_dict[col_name]
            df_i.to_csv(self.args.ck_path + "df_{}.csv".format(col_name))

            y_pred = np.array(df_i["yhat"])
            y_true = np.array(df_true[col_name])

            if len(np.unique(y_true)) < 2:
                # the same y_ture (e.g., all the empty value) cannot calculate the pearsonr
                print("\ty_true is invalid input, skipping...")
                continue

            rmses.append(np.sqrt(((y_pred - y_true) ** 2).mean()))
            maes.append(np.abs(y_pred - y_true).mean())
            corrs.append(pearsonr(y_pred, y_true)[0])
            dtws.append(dtw.distance_fast(y_pred.astype(np.double), y_true.astype(np.double)))

        # Calculate the mean and std over each stream and batch
        rmse_mean, rmse_std = np.mean(rmses), np.std(rmses)
        mae_mean, mae_std = np.mean(maes), np.std(maes)
        corr_mean, corr_std = np.mean(corrs), np.std(corrs)
        dtw_mean, dtw_std = np.mean(dtws), np.std(dtws)

        # Save Logs
        self.df_metrics.loc[0, ["test_rmse_mean", "test_mae_mean", "test_corr_mean", "test_dtw_mean",
                                "test_rmse_std", "test_mae_std", "test_corr_std", "test_dtw_std"]] = \
            rmse_mean, mae_mean, corr_mean, dtw_mean, rmse_std, mae_std, corr_std, dtw_std
        self.df_metrics.to_csv(self.args.ck_path + "metrics.csv")

        return rmse_mean, mae_mean, corr_mean, dtw_mean

    def CV_evaluate_(self, cross_valid_dict):

        # Loop the data streams (features)
        rmses, maes, corrs, dtws = [], [], [], []
        for col_name in tqdm.tqdm(cross_valid_dict.keys()):
            df_col = cross_valid_dict[col_name]
            df_col.to_csv(self.args.ck_path + "df_{}.csv".format(col_name))

            # Loop the cutoffs (batches)
            cutoff_rmses, cutoff_maes, cutoff_corrs, cutoff_dtws = [], [], [], []
            for cutoff in pd.unique(df_col["cutoff"]):
                df_i = df_col[df_col["cutoff"] == cutoff]
                y_pred = np.array(df_i["yhat"])
                y_true = np.array(df_i["y"])

                if len(np.unique(y_true)) < 2:
                    # the same y_ture (e.g., all the empty value) cannot calculate the pearsonr
                    print("\ty_true is invalid input, skipping...")
                    continue

                cutoff_rmses.append(np.sqrt(((y_pred - y_true) ** 2).mean()))
                cutoff_maes.append(np.abs(y_pred - y_true).mean())
                cutoff_corrs.append(pearsonr(y_pred, y_true)[0])
                cutoff_dtws.append(dtw.distance_fast(y_pred.astype(np.double), y_true.astype(np.double)))

            if len(cutoff_rmses) < 2:
                # Sometimes we will skip too much cutoffs, leading to empty list
                print("\t[feature] y_true is invalid input, skipping...")
                continue

            rmses.append(np.mean(cutoff_rmses))
            maes.append(np.mean(cutoff_maes))
            corrs.append(np.mean(cutoff_corrs))
            dtws.append(np.mean(cutoff_dtws))

        # Calculate the mean and std over each stream and batch
        rmse_mean, rmse_std = np.mean(rmses), np.std(rmses)
        mae_mean, mae_std = np.mean(maes), np.std(maes)
        corr_mean, corr_std = np.mean(corrs), np.std(corrs)
        dtw_mean, dtw_std = np.mean(dtws), np.std(dtws)
        """
        # Calculate the metrics 1
        col_names = cross_valid_dict.keys()
        y_preds, y_trues = [], []
        for col_name in col_names:
            df_i = cross_valid_dict[col_name]
            df_i.to_csv(self.args.ck_path + "df_{}.csv".format(col_name))

            y_pred = list(df_i["yhat"])
            y_true = list(df_i["y"])
            y_preds.append(y_pred)
            y_trues.append(y_true)
        y_preds = np.array(y_preds).T           # (samples, dim)
        y_trues = np.array(y_trues).T           # (samples, dim)

        rmses, maes, corrs, dtws = [], [], [], []
        for i in range(y_preds.shape[-1]):
            # pearsonr returns (statistic, pvalue), where we only need the first one
            rmses.append(np.sqrt(((y_preds[:, i] - y_trues[:, i]) ** 2).mean()))
            maes.append(np.abs(y_preds[:, i] - y_trues[:, i]).mean())
            corrs.append(pearsonr(y_preds[:, i], y_trues[:, i])[0])
            dtws.append(0)
            # dtws.append(dtw.distance(y_preds[:, i], y_trues[:, i]))
        rmse_mean = np.mean(rmses)
        mae_mean = np.mean(maes)
        corr_mean = np.mean(corrs)
        dtws_mean = np.mean(dtws)
        print("m1-rmse: ", rmse_mean)
        print("m1-mae: ", mae_mean)
        
        # Calculate the metrics 2
        prmses, pmaes = [], []
        for col_name in cross_valid_dict.keys():
            df_perf = performance_metrics(cross_valid_dict[col_name])
            prmses.append(df_perf['rmse'].mean())
            pmaes.append(df_perf['mae'].mean())
        prmse_mean = np.mean(prmses)
        pmae_mean = np.mean(pmaes)
        print("m2-rmse: ", prmse_mean)
        print("m2-mae: ", pmae_mean)
        """

        # Save Logs
        self.df_metrics.loc[0, ["test_rmse_mean", "test_mae_mean", "test_corr_mean", "test_dtw_mean",
                                "test_rmse_std", "test_mae_std", "test_corr_std", "test_dtw_std"]] = \
            rmse_mean, mae_mean, corr_mean, dtw_mean, rmse_std, mae_std, corr_std, dtw_std
        self.df_metrics.to_csv(self.args.ck_path + "metrics.csv")

        return rmse_mean, mae_mean, corr_mean, dtw_mean
