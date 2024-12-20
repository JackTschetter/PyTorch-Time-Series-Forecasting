import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class TCN(nn.Module):
    def __init__(self, args):
        super(TCN, self).__init__()
        self.args = args

        num_channels = [args.hidCNN]
        self.tcn_layer = TemporalConvNet(args.data_dim, num_channels)
        self.decoder = nn.Sequential(nn.Linear(args.hidRNN * args.hist_window, 128), nn.SELU(),
                                     nn.Linear(128, args.pred_window * args.data_dim))

    def forward(self, x):
        num_samples = x.shape[0]
        y = self.tcn_layer(x.transpose(1, 2))
        y = y.transpose(1, 2).reshape(num_samples, -1)
        out = self.decoder(y)
        out = out.view(num_samples, self.args.pred_window, self.args.data_dim)
        return out


class LE_TCN(nn.Module):
    def __init__(self, args, local_data_dim):
        super(LE_TCN, self).__init__()
        self.args = args

        num_channels = [args.hidCNN]
        self.tcn_layer = TemporalConvNet(local_data_dim, num_channels)
        self.decoder = nn.Linear(args.hidRNN*args.hist_window, args.local_emb_dim)

    def forward(self, x):
        num_samples = x.shape[0]
        y = self.tcn_layer(x.transpose(1, 2))
        y = y.transpose(1, 2).reshape(num_samples, -1)
        out = self.decoder(y)
        # out = out.view(num_samples, self.args.pred_window, self.args.data_dim)
        return out



############################################
############# Support Function #############
############################################
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
