import torch
from models import BaseVAE
from torch import nn
from torch.nn import functional as F
from .types_ import *


class ConCVAE(BaseVAE):

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 img_size:int = 128,
                 **kwargs) -> None:
        super(ConCVAE, self).__init__()

        self.latent_dim = latent_dim
        self.img_size = img_size
        self.embed_data = nn.Conv2d(in_channels , in_channels, kernel_size=1)
        self.embed_class = nn.Conv2d(in_channels , in_channels*2, kernel_size=1)
        
        self.hidden_dims = hidden_dims
        modules = []
        if self.hidden_dims is None:
            self.hidden_dims = [32, 64, 128, 256, 512]

        in_channels = in_channels*2 # To account for the extra label channel
        
        # Build Encoder
        for h_dim in self.hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim
        self.encoder = nn.Sequential(*modules)
        self.tensor_size = (self.img_size // (2**len(self.hidden_dims)))

        self.fc_mu = nn.Sequential(
                    nn.Conv2d(self.hidden_dims[-1], out_channels= self.latent_dim//2,
                              kernel_size= 3, padding  = 1),
                    nn.BatchNorm2d(self.latent_dim//2)
                   )
        self.fc_var = nn.Sequential(
                    nn.Conv2d(self.hidden_dims[-1], out_channels= self.latent_dim//2,
                              kernel_size= 3,  padding  = 1),
                    nn.BatchNorm2d(self.latent_dim//2),
                    nn.Softplus())

        self.decode_fc_mu = nn.Sequential(
                    nn.Conv2d(3, out_channels= 3,
                              kernel_size= 3, padding  = 1),
                    nn.BatchNorm2d(3)
                   )
        self.decode_fc_var = nn.Sequential(
                    nn.Conv2d(3, out_channels= 3,
                              kernel_size= 3,  padding  = 1),
                    nn.BatchNorm2d(3),
                    nn.Softplus())

        self.decode_class = nn.Sequential(
                    nn.Conv2d(self.hidden_dims[-1], out_channels= self.latent_dim//2,
                              kernel_size= 3, padding  = 1),
                    nn.BatchNorm2d(self.latent_dim//2),
                    nn.LeakyReLU()
                   )


        # Build Decoder
        modules = []
        self.decoder_input = nn.ConvTranspose2d(self.latent_dim, self.hidden_dims[-1], kernel_size=3, padding=1)
                    
        self.hidden_dims.reverse()
        for i in range(len(self.hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(self.hidden_dims[i],
                                       self.hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(self.hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(self.hidden_dims[-1],
                                               self.hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(self.hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(self.hidden_dims[-1], 
                                      out_channels= 3,
                                      kernel_size= 3, 
                                      padding= 1),
                            nn.Tanh())

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        # result = torch.flatten(result, start_dim=1)
        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result) + 1e-8
        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        result = self.decoder_input(z)
        # result = result.view(-1, self.hidden_dims[0], self.tensor_size, self.tensor_size)
        result = self.decoder(result)
        result = self.final_layer(result)
        mu = self.decode_fc_mu(result)
        log_var = self.decode_fc_var(result) + 1e-8
        return [mu, log_var]

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        y = kwargs['labels']
        embedded_input = self.embed_data(input)
        embedded_condition = self.embed_data(y)
        embedded_class = self.embed_class(y)
        x = torch.cat([embedded_input, embedded_condition], dim = 1)
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        z_y = self.encoder(embedded_class)
        z_y = self.decode_class(z_y)
        z = torch.cat([z, z_y], dim = 1)
        decode_mu, decode_log_var = self.decode(z)
        result = self.reparameterize(decode_mu, decode_log_var)
        return  [result, input, mu, log_var, decode_mu, decode_log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset
        recons_loss =F.mse_loss(recons, input)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp()))

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':-kld_loss}

    def sample(self,
               num_samples:int,
               current_device: int,
               **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        y = kwargs['labels'].float()
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        z = torch.cat([z, y], dim=1)
        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x, **kwargs)[0]