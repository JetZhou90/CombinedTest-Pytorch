import torch
from models import BaseVAE
from torch import nn
from torch.nn import functional as F
from .types_ import *
from torchvision import models

class JetNet(nn.Module):
    def __init__(self, latent_dim=128, in_channels = 3, image_size=128, num_class=3, backbone='mobilenet_v2',**kwargs):
        super(JetNet, self).__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.embeding_layer = nn.Conv2d(in_channels=in_channels*2,out_channels= in_channels, kernel_size=1)
        self.backbone = backbone
        if self.backbone == 'mobilenet_v2':
            # mobile_net_v2_features
            self.feature_layer = models.mobilenet_v2(pretrained=True).features
        else:
            # deeplab_v3_backbone
            self.feature_layer = models.segmentation.deeplabv3_resnet101(pretrained=True).backbone
        self.filters = 1280 if self.backbone == 'mobilenet_v2' else 2048
        self.div = 32 if self.backbone == 'mobilenet_v2' else 8 
        self.embeding_class = nn.Conv2d(in_channels=in_channels,out_channels= self.latent_dim//2, kernel_size=1,stride=self.div)
        self.new_image_size = int(self.image_size / self.div)+1 if self.image_size / self.div > int(self.image_size / self.div) else int(self.image_size / self.div)
        self.classfier = nn.Sequential(
            nn.Linear(in_features=self.filters*self.new_image_size*self.new_image_size, out_features=512),
            nn.ReLU6(),
            nn.Linear(512, 128),
            nn.ReLU6(),
            nn.Linear(128, 32),
            nn.ReLU6(),
            nn.Linear(32,num_class),
            nn.Softmax()
        )
        self.mu = nn.Sequential(nn.Conv2d(self.filters, out_channels= self.latent_dim//2, kernel_size= 3, padding  = 1),
                                nn.BatchNorm2d(self.latent_dim//2))
        self.var = nn.Sequential(nn.Conv2d(self.filters, out_channels= self.latent_dim//2,kernel_size= 3, padding= 1),
                                 nn.BatchNorm2d(self.latent_dim//2),nn.Softplus())
        hidden_dims = [512, 256, 128, 64, ] if self.backbone == 'mobilenet_v2' else [256, 64,]
        self.decoder_input = nn.ConvTranspose2d(self.latent_dim, hidden_dims[0], kernel_size=3, padding=1)
        modules = []
        in_channels = hidden_dims[0]
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1,output_padding=1,),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim
            
        self.generator = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               padding=1,
                                               stride=2,
                                               output_padding=1,
                                               ),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], 
                                      out_channels= 3,
                                      kernel_size= 3, 
                                      padding= 1),
                            nn.Tanh())
        
    def encode(self, input: Tensor) -> List[Tensor]:
        if self.backbone == 'mobilenet_v2':
            result = self.feature_layer(input)
        else:
            result = self.feature_layer(input)['out']
        mu = self.mu(result)
        log_var = self.var(result) + 1e-8
        return [mu, log_var]
    
    def decode(self, z: Tensor) -> Tensor:
        z = self.decoder_input(z)
        result = self.generator(z)
        result = self.final_layer(result)
        return result
    
    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std) 
        return eps * std + mu
    
    def match_class(self, input:Tensor, condition:Tensor) -> Tensor:
        if self.backbone == 'mobilenet_v2':
            feature_i = self.feature_layer(input)
            feature_c = self.feature_layer(condition)
        else:
            feature_i = self.feature_layer(input)['out']
            feature_c = self.feature_layer(condition)['out']
        diff = abs(feature_c-feature_i)
        diff = diff.view(-1,self.filters*self.new_image_size*self.new_image_size)
        class_label = self.classfier(diff)
        return class_label

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        condition = kwargs['labels']
        concat_x = torch.cat([input, condition], dim=1)
        class_label = self.match_class(input,condition)
        embedded_input = self.embeding_layer(concat_x)
        mu, log_var = self.encode(embedded_input)
        z = self.reparameterize(mu, log_var)
        embedding_condition = self.embeding_class(condition)
        concat_z = torch.cat([z, embedding_condition], dim=1)
        fake = self.decode(concat_z)
        return [fake, input, mu, log_var, class_label]
    
    def loss_function(self, *args,**kwargs) -> dict:
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        class_y = args[4]
        true_y = kwargs['true_label']
        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset
        recons_loss =F.mse_loss(recons, input)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp()))
        class_loss = F.cross_entropy(class_y, true_y)
        loss = recons_loss + kld_weight * kld_loss + class_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':-kld_loss, 'class_cross_entropy':class_loss}
