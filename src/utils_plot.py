
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


def draw_metric_curves(file_path, col_names, fig_name):
    """
    Draw curves corresponding to [col_names] of single file [file_path].
    """

    df_metrics = pd.read_csv(file_path+"metrics.csv")

    if len(df_metrics) == 1:
        print("Skip the plot generation since only one epoch.")
        return
    if np.all(np.isnan(df_metrics.values.astype(float))):
        print("Skip the plot generation since no valid values.")
        return
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(7, 5))
        for col_name in col_names:
            df_metrics[col_name].dropna().plot(ax=ax1, alpha=0.7, grid=True)
        # ax1.set_yticks(np.linspace(start=0, stop=0.4, num=9))
        # ax1.set_ylim(-0.01, 0.5)
        ax1.set_title(fig_name)
        ax1.legend(fontsize='xx-large')
        fig.savefig(file_path + "{}.png".format(fig_name), dpi=100, transparent=True, bbox_inches="tight")
        # fig.show()

# draw_metric_curves(file_path="./checkpoint_0825-1416-43/", col_names=["train_loss", "valid_rse", "test_rse"], fig_name="Metrics")
# or ["val_corr", "test_corr"] or ["train_loss", "valid_rse", "test_rse"]


def draw_metric_curves_compare(file_path_1, file_path_2, col_names):
    """
    Draw curves corresponding to [col_names] of single file [file_path_1] and [file_path_2].
    """

    df_metrics_1 = pd.read_csv(file_path_1 + "metrics.csv")
    df_metrics_2 = pd.read_csv(file_path_2 + "metrics.csv")
    df_metrics_1.columns = "single_" + df_metrics_1.columns
    df_metrics_2.columns = "dist_" + df_metrics_2.columns
    df_metrics = pd.concat([df_metrics_1, df_metrics_2])

    col_names = ["single_"+name for name in col_names] + ["dist_"+name for name in col_names]

    fig, ax1 = plt.subplots(1, 1, figsize=(7, 5))
    for col_name in col_names:
        df_metrics[col_name].dropna().plot(ax=ax1, alpha=0.7, grid=True)
    # ax1.set_yticks(np.linspace(start=0, stop=0.4, num=9))
    # ax1.set_ylim(-0.01, 0.5)
    ax1.set_title("RSE on test set")
    ax1.legend(['single_test_rse', 'dist_test_rse'])  # ['single_test_loss', 'dist_test_loss']
    fig.savefig(file_path_1 + "Summary_loss_compare.png", dpi=100, transparent=True, bbox_inches="tight")
    # fig.show()


# draw_metric_curves_compare(file_path_1="./checkpoint_0825-2013-56/", file_path_2="./checkpoint_0825-2012-18/", col_names=["valid_rse"])


def draw_y_curve_hour(args, myModel, config):
    """
    Draw y_pred and y_true curve: [batches, predict time, data features]
        [:N, j, k] -> For the first consecutive N samples, plot the future j-th hour of k-th data feature (time streaming).
    """
    j, k = config[0], config[1]

    plt.plot(myModel.y_true[:24*7, j, k], label='Real')
    plt.plot(myModel.y_pred[:24*7, j, k], label='Pred')
    plt.legend(fontsize='xx-large', ncol=2)
    plt.title(args.model)
    plt.ylabel('Scaled Traffic')
    plt.xlabel('Time')
    plt.savefig(args.ck_path + "{}_nodes{}_hist{}_pred{}_j{}_k{}.png".format(args.model, args.n_local, args.hist_window, args.pred_window, j, k),
                dpi=100, transparent=True, bbox_inches="tight")
    # plt.show()
    plt.cla()


def draw_y_curve_sample(args, myModel, config):
    """
    Draw y_pred and y_true curve: [batches, predict time, data features]
        [i, :, k] -> For the i-th sample, plot the future (all) hours of k-th data feature (time streaming).
    """
    i, k = config[0], config[1]

    plt.plot(myModel.y_true[i, :, k], label='Real')
    plt.plot(myModel.y_pred[i, :, k], label='Pred')
    plt.legend(fontsize='xx-large', ncol=2)
    plt.title(args.model)
    plt.ylabel('Scaled Traffic')
    plt.xlabel('Time')
    plt.savefig(args.ck_path + "{}_nodes{}_hist{}_pred{}_i{}_k{}.png".format(args.model, args.n_local, args.hist_window, args.pred_window, i, k),
                dpi=100, transparent=True, bbox_inches="tight")
    # plt.show()
    plt.cla()


class suppress_stdout_stderr(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    '''

    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        for fd in self.null_fds + self.save_fds:
            os.close(fd)
