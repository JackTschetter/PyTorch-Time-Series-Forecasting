
##########################################
### Handle nan value in the results ######
##########################################

import numpy as np
from scipy.stats import pearsonr
import pandas as pd
from dtaidistance import dtw
import os
import tqdm

path = "./checkpoint_DATE-0125-0951-09-MODEL-Prophet-DATA-electricity-HIST-36-PRED-168/"
files = os.listdir(path)

# Loop the data streams (features)
rmses, maes, corrs, dtws = [], [], [], []
for i, file in enumerate(tqdm.tqdm(files)):
    if "df_" not in file:
        continue

    df_col = pd.read_csv(path+file)

    # Loop the cutoffs (batches)
    cutoff_rmses, cutoff_maes, cutoff_corrs, cutoff_dtws = [], [], [], []
    for cutoff in pd.unique(df_col["cutoff"]):
        df_i = df_col[df_col["cutoff"] == cutoff]
        y_pred = np.array(df_i["yhat"])
        y_true = np.array(df_i["y"])

        if len(np.unique(y_true)) < 2:
            # the same y_ture (e.g., all the empty value) cannot calculate the pearsonr
            print("\t[cutoff] y_true is invalid input, skipping...")
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
# Find the location of nan: np.argwhere(np.isnan(corrs))
rmse_mean, rmse_std = np.mean(rmses), np.std(rmses)
mae_mean, mae_std = np.mean(maes), np.std(maes)
corr_mean, corr_std = np.mean(corrs), np.std(corrs)
dtw_mean, dtw_std = np.mean(dtws), np.std(dtws)

print("test rmse mean {:5.6f} | test mae mean {:5.6f} | test corr mean {:5.6f} | test dtw mean {:5.6f}".format(rmse_mean, mae_mean, corr_mean, dtw_mean))
print("test rmse std {:5.6f} | test mae std {:5.6f} | test corr std {:5.6f} | test dtw std {:5.6f}".format(rmse_std, mae_std, corr_std, dtw_std))
