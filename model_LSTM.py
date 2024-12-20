
import torch
import torch.nn as nn
import torch.nn.functional as F

############################################
################### LSTM ###################
############################################
class LSTM(nn.Module):
    def __init__(self, args):
        super(LSTM, self).__init__()
        self.args = args

        self.lstm_layer = nn.LSTM(input_size=args.data_dim, hidden_size=args.hidRNN, num_layers=2)
        self.mlp = nn.Sequential(nn.Linear(args.hidRNN*args.hist_window, 128), nn.SELU(),
                                 nn.Linear(128, args.pred_window*args.data_dim))

    def forward(self, x):
        num_samples = len(x)
        x, _ = self.lstm_layer(x)
        x = x.reshape(num_samples, -1)
        x = self.mlp(x)
        x = x.view(num_samples, self.args.pred_window, self.args.data_dim)
        return x


class LE_LSTM(nn.Module):
    def __init__(self, args, local_data_dim, mode="multi"):
        super(LE_LSTM, self).__init__()
        self.args = args
        self.mode = mode
        self.local_data_dim = local_data_dim

        self.lstm_layer = nn.LSTM(input_size=local_data_dim, hidden_size=args.hidRNN, num_layers=2)
        if mode == "uni":
            # Without global model, output is uni-variate prediction
            self.mlp = nn.Sequential(nn.Linear(args.hidRNN * args.hist_window, 128),
                                     nn.Linear(128, args.pred_window*local_data_dim))
        else:
            # With global model, output is local embedding for multi-variate prediction in global
            self.mlp = nn.Sequential(nn.Linear(args.hidRNN*args.hist_window, args.local_emb_dim))

    def forward(self, x):
        num_samples = len(x)
        x, _ = self.lstm_layer(x)
        x = x.reshape(num_samples, -1)
        x = self.mlp(x)
        if self.mode == "uni":
            x = x.view(num_samples, self.args.pred_window, self.local_data_dim)
        return x


##############################################
################### LSTNet ###################
##############################################
class LSTNet(nn.Module):
    def __init__(self, args):
        super(LSTNet, self).__init__()
        self.hist_window = args.hist_window
        self.pred_window = args.pred_window
        self.m = args.data_dim
        self.hidR = args.hidRNN
        self.hidC = args.hidCNN
        self.hidS = args.hidSkip
        self.Ck = args.CNN_kernel
        self.skip = args.skip
        self.pt = int((self.hist_window - self.Ck)/self.skip)
        self.hw = args.highway_window

        self.conv1 = nn.Conv2d(1, self.hidC, kernel_size=(self.Ck, self.m))
        self.GRU1 = nn.GRU(self.hidC, self.hidR)
        self.dropout = nn.Dropout(p=args.dropout)

        if self.skip > 0:
            self.GRUskip = nn.GRU(self.hidC, self.hidS)
            self.linear1 = nn.Sequential(nn.Linear(self.hidR + self.skip * self.hidS, 128), nn.SELU(),
                                         nn.Linear(128, self.pred_window*self.m))
        else:
            self.linear1 = nn.Linear(self.hidR, self.pred_window*self.m)

        if self.hw > 0:
            self.highway = nn.Linear(self.hw, self.pred_window)

    def forward(self, x):
        batch_size = x.size(0)

        # CNN
        c = x.view(-1, 1, self.hist_window, self.m)
        c = F.relu(self.conv1(c))
        c = self.dropout(c)
        c = torch.squeeze(c, 3)

        # RNN
        r = c.permute(2, 0, 1).contiguous()
        _, r = self.GRU1(r)
        r = self.dropout(torch.squeeze(r, 0))

        # skip-RNN
        if self.skip > 0:
            s = c[:, :, int(-self.pt * self.skip):].contiguous()
            s = s.view(batch_size, self.hidC, self.pt, self.skip)
            s = s.permute(2, 0, 3, 1).contiguous()
            s = s.view(self.pt, batch_size * self.skip, self.hidC)
            _, s = self.GRUskip(s)
            s = s.view(batch_size, self.skip * self.hidS)
            s = self.dropout(s)
            r = torch.cat((r, s), 1)

        res = self.linear1(r)
        res = res.view(batch_size, self.pred_window, self.m)

        # highway
        if self.hw > 0:
            z = x[:, -self.hw:, :]
            z = z.permute(0, 2, 1).contiguous().view(-1, self.hw)
            z = self.highway(z)
            z = z.view(batch_size, self.pred_window, self.m)
            res = res + z

        return res


class LE_LSTNet(nn.Module):
    def __init__(self, args, local_data_dim):
        super(LE_LSTNet, self).__init__()
        self.hist_window = args.hist_window
        self.pred_window = args.pred_window
        self.m = local_data_dim  # int(args.data_dim/args.n_local)
        self.hidR = args.hidRNN
        self.hidC = args.hidCNN
        self.hidS = args.hidSkip
        self.Ck = args.CNN_kernel
        self.skip = args.skip
        self.pt = int((self.hist_window - self.Ck)/self.skip)
        self.hw = args.highway_window
        self.local_emb_dim = args.local_emb_dim

        self.conv1 = nn.Conv2d(1, self.hidC, kernel_size=(self.Ck, self.m))
        self.GRU1 = nn.GRU(self.hidC, self.hidR)
        self.dropout = nn.Dropout(p=args.dropout)

        if self.skip > 0:
            self.GRUskip = nn.GRU(self.hidC, self.hidS)
            self.linear1 = nn.Linear(self.hidR + self.skip * self.hidS, self.local_emb_dim)  # self.pred_window*self.m
        else:
            self.linear1 = nn.Linear(self.hidR, self.local_emb_dim)  # self.pred_window*self.m

        if self.hw > 0:
            self.highway = nn.Linear(self.hw, self.local_emb_dim)

    def forward(self, x):
        batch_size = x.size(0)

        # CNN
        c = x.view(-1, 1, self.hist_window, self.m)
        c = F.relu(self.conv1(c))
        c = self.dropout(c)
        c = torch.squeeze(c, 3)

        # RNN
        r = c.permute(2, 0, 1).contiguous()
        _, r = self.GRU1(r)
        r = self.dropout(torch.squeeze(r, 0))

        # skip-rnn
        if self.skip > 0:
            s = c[:, :, int(-self.pt * self.skip):].contiguous()
            s = s.view(batch_size, self.hidC, self.pt, self.skip)
            s = s.permute(2, 0, 3, 1).contiguous()
            s = s.view(self.pt, batch_size * self.skip, self.hidC)
            _, s = self.GRUskip(s)
            s = s.view(batch_size, self.skip * self.hidS)
            s = self.dropout(s)
            r = torch.cat((r, s), 1)

        res = self.linear1(r)
        # res = res.view(batch_size, self.pred_window, self.m)

        # highway
        # if self.hw > 0:
        #     z = x[:, -self.hw:, :]
        #     z = z.permute(0, 2, 1).contiguous().view(-1, self.hw)
        #     z = self.highway(z)
        #     z = z.view(batch_size, self.local_emb_dim)
        #     res = res + z

        return res

