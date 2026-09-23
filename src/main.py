import os
import shutil
from datetime import datetime
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from model_LSTM import LSTM, LE_LSTM, LSTNet, LE_LSTNet
from model_TCN import TCN, LE_TCN
from model_INFORMER import Informer

from wrapper_LSTM import myLSTM
from wrapper_DIVIDE import myDIVIDE, GE_MLP
from wrapper_UnivarDIVIDE import myUnivarDIVIDE
from wrapper_INFORMER import myInformer

from utils_data import Data_utility
from utils_plot import draw_metric_curves, draw_y_curve_hour, draw_y_curve_sample

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

##########################################################
# Hyper-parameters
##########################################################
args_dict = {}
parser = argparse.ArgumentParser(description='PyTorch Time series forecasting')
# Main setting
parser.add_argument('--data',           type=str,   default='wiki_small',     help='location of the data file', choices=["wiki_small"])
parser.add_argument('--model',          type=str,   default='LSTM',          help='model name',                choices=["Prophet",
                                                                                                                          "LSTM", "TCN", "LSTNet",
                                                                                                                          "UnivarDIVIDE", "DIVIDE",
                                                                                                                          "Informer", "INFORMER"])
# Environment setting
parser.add_argument('--gpu_id',         type=int,   default=2,                  help='GPU id')
parser.add_argument('--ck_path',        type=str,   default='./checkpoint_',    help='path to save the final model')  # './checkpoint_'
parser.add_argument('--seed',           type=int,   default=54321,              help='random seed')
parser.add_argument('--model_path',     type=str)
# Dataset setting
parser.add_argument('--hist_window',    type=int,   default=36,                 help='history window size')
parser.add_argument('--pred_window',    type=int,   default=24,                 help='future window size')
parser.add_argument('--normalize',      type=int,   default=1)
parser.add_argument('--data_dim',       type=int,   default=4)
# Learning setting
parser.add_argument('--n_local',        type=int,   default=4,                  help='number of local nodes')
parser.add_argument('--lr',             type=float, default=0.01,               help='learning rate')
parser.add_argument('--L1Loss',         type=bool,  default=True,               help='whether use L1 loss for training')
parser.add_argument('--epochs',         type=int,   default=10,                help='upper epoch limit') #Changed from 125 to 10.
parser.add_argument('--batch_size',     type=int,   default=128,                help='batch size')
# Model setting
parser.add_argument('--hidCNN',         type=int,   default=128,                help='number of CNN hidden units')
parser.add_argument('--hidRNN',         type=int,   default=128,                help='number of RNN hidden units')
parser.add_argument('--CNN_kernel',     type=int,   default=6,                  help='the kernel size of the CNN layers')
parser.add_argument('--highway_window', type=int,   default=24,                 help='The window size of the highway component')
parser.add_argument('--clip',           type=float, default=10.,                help='gradient clipping')
parser.add_argument('--dropout',        type=float, default=0.2,                help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--skip',           type=float, default=6)
parser.add_argument('--hidSkip',        type=int,   default=5)
parser.add_argument('--local_emb_dim',  type=int,   default=64)
# Informer setting
parser.add_argument('--label_len',      type=int,   default=18,                 help='Informer decoder context length')
parser.add_argument('--factor',         type=int,   default=5,                  help='Informer ProbSparse attention factor')
parser.add_argument('--d_model',        type=int,   default=64,                 help='Informer hidden dimension')
parser.add_argument('--n_heads',        type=int,   default=4,                  help='Informer number of attention heads')
parser.add_argument('--e_layers',       type=int,   default=2,                  help='Informer encoder layers')
parser.add_argument('--d_layers',       type=int,   default=1,                  help='Informer decoder layers')
parser.add_argument('--d_ff',           type=int,   default=128,                help='Informer feed-forward dimension')
parser.add_argument('--attn',           type=str,   default='prob',             help='Informer attention type', choices=['prob', 'full'])
parser.add_argument('--embed',          type=str,   default='fixed',            help='Informer time embedding type', choices=['fixed'])
parser.add_argument('--freq',           type=str,   default='h',                help='Informer timestamp frequency', choices=['h'])
args = parser.parse_args()


##########################################################
# Set environment
##########################################################
# Set GPU ID
device = torch.device('cuda:'+str(args.gpu_id) if torch.cuda.is_available() else 'cpu')

# Set the random seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# Set checkpoint folder
if len(args.ck_path) < 15:
    # Option 1: create a new folder and retrain the model
    FLAG_NEW_TRAIN = True
    checkpoint_path = args.ck_path + "DATE-{}{}-{}{}-{}-MODEL-{}-DATA-{}-HIST-{}-PRED-{}/"\
        .format(datetime.now().strftime("%m"), datetime.now().strftime("%d"), datetime.now().strftime("%H"),
                datetime.now().strftime("%M"), datetime.now().strftime("%S"),
                args.model, args.data, args.hist_window, args.pred_window)
    # Create folder
    os.mkdir(checkpoint_path)
    args.model_path = checkpoint_path
    args.ck_path = checkpoint_path
    # Save args
    args_dict = vars(args)
    with open(os.path.join(checkpoint_path, "args.txt"), 'w') as f:
        for key, value in args_dict.items():
            f.write('%s:%s\n' % (key, value))

elif "MODEL-Prophet-DATA" in args.ck_path:
    # Option 2: create a new folder but load the pre-trained model
    FLAG_NEW_TRAIN = False
    checkpoint_path = "./checkpoint_DATE-{}{}-{}{}-{}-MODEL-{}-DATA-{}-HIST-{}-PRED-{}/" \
        .format(datetime.now().strftime("%m"), datetime.now().strftime("%d"), datetime.now().strftime("%H"),
                datetime.now().strftime("%M"), datetime.now().strftime("%S"),
                args.model, args.data, args.hist_window, args.pred_window)
    # Create a new folder
    # shutil.copytree(src=args.ck_path, dst=checkpoint_path)
    os.mkdir(checkpoint_path)
    args.model_path = args.ck_path
    args.ck_path = checkpoint_path
    # Save args
    args_dict = vars(args)
    with open(os.path.join(checkpoint_path, "args.txt"), 'w') as f:
        for key, value in args_dict.items():
            f.write('%s:%s\n' % (key, value))

else:
    # Option 3: use the old folder and load the pre-trained model
    FLAG_NEW_TRAIN = False
    args.model_path = args.ck_path


##########################################################
# Load data
##########################################################
print('Loading data ...')
Data = Data_utility(file_name=args.data, train=0.5, valid=0.2, device=device,
                    pred_window=args.pred_window, hist_window=args.hist_window, normalize=args.normalize)
args.data_dim = Data.m


##########################################################
# Model
##########################################################
if args.model == "Prophet":
    from model_MultiProphet import MultiProphet
    from wrapper_Prophet import myProphet

    FLAG_CROSS_VALID = True

    # Model
    col_names = list(Data.rawdat_df.columns)
    col_names.remove('ds')
    model = MultiProphet(columns=col_names, uncertainty_samples=0)  # uncertainty_samples=0 can speed up

    # Wrap up
    myModel = myProphet(args, model, FLAG_CROSS_VALID)

elif args.model in ["LSTM", "TCN", "LSTNet"]:
    # Model
    if args.model == "LSTNet":
        model = LSTNet(args).to(device)
    elif args.model == "LSTM":
        model = LSTM(args).to(device)
    elif args.model == "TCN":
        model = TCN(args).to(device)

    # Optimizer
    # optim = Optim(model.parameters(), args.optim, args.lr, args.clip,)
    optim = optim.Adam(model.parameters())

    # Loss
    if args.L1Loss:
        criterion = nn.L1Loss().to(device)
    else:
        criterion = nn.MSELoss().to(device)
    evaluateL1 = nn.L1Loss(reduction="none").to(device)
    evaluateL2 = nn.MSELoss(reduction="none").to(device)

    # Wrap up
    myModel = myLSTM(args, model, optim, criterion, evaluateL1, evaluateL2)

elif args.model == "UnivarDIVIDE":
    # Models
    data_chunks = np.array_split(np.arange(args.data_dim), args.n_local)
    models = [LE_LSTM(args, len(data_chunks[k]), mode="uni").to(device) for k in range(args.n_local)]

    # Optimizer
    optims = []
    for i in range(args.n_local):
        optims.append(optim.Adam(models[i].parameters(), lr=args.lr))

    # Loss
    if args.L1Loss:
        criterion = nn.L1Loss().to(device)
    else:
        criterion = nn.MSELoss().to(device)
    evaluateL1 = nn.L1Loss(reduction="none").to(device)
    evaluateL2 = nn.MSELoss(reduction="none").to(device)

    # Wrap up
    myModel = myUnivarDIVIDE(args, models, optims, criterion, evaluateL1, evaluateL2)

elif args.model == "DIVIDE":
    # Models
    global_model = [GE_MLP(args).to(device)]

    data_chunks = np.array_split(np.arange(args.data_dim), args.n_local)
    local_models = [LE_LSTM(args, len(data_chunks[k])).to(device) for k in range(args.n_local)]
    models = global_model + local_models

    # Optimizer
    optims = []
    for i in range(1+args.n_local):
        optims.append(optim.Adam(models[i].parameters(), lr=args.lr))

    # Loss
    if args.L1Loss:
        criterion = nn.L1Loss().to(device)
    else:
        criterion = nn.MSELoss().to(device)
    evaluateL1 = nn.L1Loss(reduction="none").to(device)
    evaluateL2 = nn.MSELoss(reduction="none").to(device)

    # Wrap up
    myModel = myDIVIDE(args, models, optims, criterion, evaluateL1, evaluateL2)

elif args.model in ["Informer", "INFORMER"]:
    if args.label_len > args.hist_window:
        raise ValueError("--label_len must be less than or equal to --hist_window for Informer.")

    model = Informer(
        enc_in=args.data_dim,
        dec_in=args.data_dim,
        c_out=args.data_dim,
        seq_len=args.hist_window,
        label_len=args.label_len,
        out_len=args.pred_window,
        factor=args.factor,
        d_model=args.d_model,
        n_heads=args.n_heads,
        e_layers=args.e_layers,
        d_layers=args.d_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        attn=args.attn,
        embed=args.embed,
        freq=args.freq,
        activation='gelu',
        output_attention=False,
        distil=True,
        mix=True,
        device=device,
    ).to(device)

    optim = optim.Adam(model.parameters(), lr=args.lr)

    if args.L1Loss:
        criterion = nn.L1Loss().to(device)
    else:
        criterion = nn.MSELoss().to(device)
    evaluateL1 = nn.L1Loss(reduction="none").to(device)
    evaluateL2 = nn.MSELoss(reduction="none").to(device)

    myModel = myInformer(args, model, optim, criterion, evaluateL1, evaluateL2)
##########################################################
# Train
##########################################################
if FLAG_NEW_TRAIN:
    myModel.train(Data)


##########################################################
# Eval and save
##########################################################
myModel.evaluate(Data)


# Plot results
if args.model != "Prophet":
    for p1 in [0, 23, 71, 167, 335]:
        if p1 < args.pred_window:
            for p2 in range(myModel.y_true.shape[-1]):
                draw_y_curve_hour(args, myModel, [p1, p2])
                draw_y_curve_sample(args, myModel, [p1, p2])

                if p2 > 3:
                    break

    draw_metric_curves(file_path=args.ck_path, col_names=["valid_rmse_mean", "test_mae_mean"], fig_name="Metrics")


# Finished
print("Done!")
