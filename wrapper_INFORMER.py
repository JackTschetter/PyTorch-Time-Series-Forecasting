import torch
import torch.nn as nn
import torch.nn.functional as F 

from Informer_Utils.masking import TriangularCausalMask, ProbMask
from Informer_Models.encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack
from Informer_Models.decoder import Decoder, DecoderLayer
from Informer_Models.attn import FullAttention, ProbAttention, AttentionLayer
from Informer_Models.embed import DataEmbedding

class myInformer():
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len, factor = 5, d_model = 512, n_heads = 8, e_layers = 3, d_layers = 2, d_ff = 512, dropout = 0.0, attn='prob', embed='fixed', freq='h', activation='gelu', output_attention = False, distill=True, mix=True, device = torch.device('cuda:0')):
        super(Informer, self).__init__()
        
        def train(self, Data):
            print('Training Informer...')
    
        def evaluate(self, Data):
            print('Evaluating Informer...')