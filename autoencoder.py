import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import math


class AutoEncoder(nn.Module):

    def __init__(self, args):
        super(AutoEncoder, self).__init__()
        self.args = args
        self.input_dim = args.input_dim
        self.output_dim = self.input_dim
        self.input_channels = args.input_channels
        self.output_channels = self.input_channels
        self.hidden_channels = args.hidden_channels
        self.kernel_sizes = args.kernel_sizes     
        self.latent_dim = args.latent_dim
        self.n_clusters = args.n_clusters
        self.latent_input_dim = self.calc_input_dim_latent()
        self.channels_list = (args.hidden_channels + 
                             args.hidden_channels[::-1])
        self.kernel_sizes_list = (args.kernel_sizes + 
                                args.kernel_sizes[::-1])
        self.n_layers = len(self.channels_list)+2 # convolutional layers X 2 + latent layer + dense layer
        self.device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')

        # Validation check
        assert self.n_layers % 2 == 0   # even number of layers

        # Encoder Network
        self.encoder_layers = OrderedDict()
        for idx, (hidden_channel, kernel_size) in enumerate(zip(self.hidden_channels, self.kernel_sizes)):
            if idx == 0:
                self.encoder_layers.update(
                    {
                        'conv0': nn.Conv2d(self.input_channels, hidden_channel, kernel_size=kernel_size, stride=2, device = self.device),
                        'activation0': nn.ReLU()   
                    }
                )
            else:
                self.encoder_layers.update(
                    {
                        'conv{}'.format(idx): nn.Conv2d(
                            self.hidden_channels[idx-1], hidden_channel, kernel_size=kernel_size, stride=2, device = self.device),
                        'activation{}'.format(idx): nn.ReLU(),
                        'bn{}'.format(idx): nn.BatchNorm2d(hidden_channel, device = self.device)
                    }
                )

        # dense latent layer of encoder
        self.latent_encoder = nn.Linear(self.latent_input_dim, self.latent_dim)
        self.bn_latent = nn.BatchNorm1d(self.latent_dim)


        # dense layer of decoder, to increase dimensions before transposed convolutions
        self.fc_decoder = nn.Linear(self.latent_dim, self.latent_input_dim)
        self.bn0 = nn.BatchNorm1d(self.latent_input_dim)

        # Decoder Network
        self.decoder_layers = OrderedDict()
        tmp_hidden_channels = self.hidden_channels[::-1]
        tmp_kernel_sizes = self.kernel_sizes[::-1]

        for idx, (hidden_channel, kernel_size) in enumerate(zip(tmp_hidden_channels, tmp_kernel_sizes)):
            if idx == len(tmp_hidden_channels) - 1:
                self.decoder_layers.update(
                    {
                        'trans_conv{}'.format(idx): nn.ConvTranspose2d(
                            hidden_channel, self.output_channels, kernel_size=kernel_size, stride=2, device = self.device)
                    }
                )
            else:
                self.decoder_layers.update(
                    {
                        'trans_conv{}'.format(idx) : nn.ConvTranspose2d(
                            hidden_channel, tmp_hidden_channels[idx+1], kernel_size=kernel_size, stride=2, device = self.device),
                        'activation{}'.format(idx): nn.ReLU(),
                        'bn{}'.format(idx): nn.BatchNorm2d(
                            tmp_hidden_channels[idx+1], device = self.device)
                    }
                )

    def calc_input_dim_latent(self):
        out = math.sqrt(self.input_dim)
        for kernel_size in self.kernel_sizes:
            out = (out-kernel_size)//2 + 1
        return int(out*out*self.hidden_channels[-1])
        
    
    def latent_encoding_layer(self, X):  
        X = X.view(-1, self.latent_input_dim)  
        X = F.relu(self.latent_encoder(X))
        X = self.bn_latent(X)
        return X

    def fc_decoding_layer(self, X):     
        decoder_input_dim = int(math.sqrt(self.latent_input_dim/self.hidden_channels[-1]))
        X = F.relu(self.fc_decoder(X))
        X = self.bn0(X)        
        X = X.reshape(-1, self.hidden_channels[-1], decoder_input_dim, decoder_input_dim)
        return X

    def encoder(self, X):
        skip_X_list = []
        for layer in self.encoder_layers:
          X = self.encoder_layers[layer](X)
          if layer[:-1] == 'conv':
            skip_X_list.append(X)   # saving convolved X at each convolution layer to be added later to transposed convolution layers' inputs
        return X, skip_X_list

    def decoder(self, X, skip_X_list):
        for layer in self.decoder_layers:
            if layer[:-1] == 'trans_conv':
                X = X + skip_X_list[len(self.hidden_channels)-1-int(layer[-1])] 
                                    # adding the convolved X to the input of the corresponding transposed convolution layers
            X = self.decoder_layers[layer](X)
        return X

    def __repr__(self):
        # repr_str = '[Structure]: {}-'.format(self.input_dim)
        # for idx, dim in enumerate(self.dims_list):
        #     repr_str += '{}-'.format(dim)
        # repr_str += str(self.output_dim) + '\n'
        # repr_str += '[n_layers]: {}'.format(self.n_layers) + '\n'
        # repr_str += '[n_clusters]: {}'.format(self.n_clusters) + '\n'
        # repr_str += '[input_dims]: {}'.format(self.input_dim)
        # return repr_str
        return None

    def __str__(self):
        return self.__repr__()

    def forward(self, X, latent=False):
        X, skip_X_list = self.encoder(X)
        output = self.latent_encoding_layer(X)
        if latent:
            return output
        output = self.fc_decoding_layer(output)
        return self.decoder(output, skip_X_list)
