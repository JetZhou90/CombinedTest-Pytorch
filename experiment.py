import math
import torch
from torch import optim
from models import BaseVAE, DetectModel, BaseUnet
from models.types_ import *
from lightning_utils import data_loader
import pytorch_lightning as pl
from torchvision import transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from dataload import *
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess, STANDARD_COLORS, standard_to_bgr, get_index_label, plot_one_box


img_path = 'test/wps002.png'

class EfficientExperiment(pl.LightningModule):

    def __init__(self, 
                detect_model: DetectModel,
                params: dict) -> None:
        super(EfficientExperiment, self).__init__()

        self.model = detect_model
        self.params = params
        self.curr_device = None
        self.hold_graph = False
        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass

    def forward(self, input):
        return self.model(input)
    
    def training_step(self, batch, batch_idx, optimizer_idx = 0):
        real_img, annotations = batch['img'] , batch['annot']
        self.curr_device = real_img.device
        _, regression, classification, anchors = self.forward(real_img)
        train_loss = self.model.loss_function(classification, regression, anchors, annotations)
        self.logger.experiment.log({key: val.item() for key, val in train_loss.items()})
        return train_loss

    def validation_step(self, batch, batch_idx, optimizer_idx = 0):
        real_img, annotations = batch['img'] , batch['annot']
        self.curr_device = real_img.device
        _, regression, classification, anchors = self.forward(real_img)
        val_loss = self.model.loss_function(classification, regression, anchors, annotations)
        self.logger.experiment.log({key: val.item() for key, val in val_loss.items()})
        return val_loss

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        tensorboard_logs = {'avg_val_loss': avg_loss}
        self.model.detect_image(img_path)
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

 
    def configure_optimizers(self):

        optims = []
        scheds = []
        optimizer = optim.AdamW(self.model.parameters(),
                               lr=self.params['LR'])
        optims.append(optimizer)
        # Check if more than 1 optimizer is required (Used for adversarial training)
        try:
            if self.params['LR_2'] is not None:
                optimizer2 = optim.AdamW(getattr(self.model,self.params['submodel']).parameters(),
                                        lr=self.params['LR_2'])
                optims.append(optimizer2)
        except:
            pass

        try:
            if self.params['scheduler_gamma'] is not None:
                scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
                                                             gamma = self.params['scheduler_gamma'])
                scheds.append(scheduler)

                # Check if another scheduler is required for the second optimizer
                try:
                    if self.params['scheduler_gamma_2'] is not None:
                        scheduler2 = optim.lr_scheduler.ExponentialLR(optims[1],
                                                                      gamma = self.params['scheduler_gamma_2'])
                        scheds.append(scheduler2)
                except:
                    pass
                return optims, scheds
        except:
            return optims



    @data_loader
    def train_dataloader(self):
        dataset = Detect_Dataset_folder(img_dir=self.params['train_data_path'],ann_dir=self.params['train_ann_path'],
        transform=transforms.Compose([Normalizer(mean=self.params['mean'], std=self.params['std']), Augmenter(),Resizer(self.params['input_size'])]))
            # raise ValueError('Undefined dataset type')
        self.num_train_imgs = len(dataset)
        training_params = {
            'batch_size': self.params['batch_size'],
            'shuffle': True,
            'drop_last': True,
            'collate_fn': collater,
            'num_workers': self.params['num_workers']}

        return DataLoader(dataset, **training_params)

    @data_loader
    def val_dataloader(self):
        self.sample_dataloader =Detect_Dataset_folder(img_dir=self.params['train_data_path'],ann_dir=self.params['train_ann_path'],
        transform=transforms.Compose([Normalizer(mean=self.params['mean'], std=self.params['std']), Augmenter(),Resizer(self.params['input_size'])]))
            # raise ValueError('Undefined dataset type')
        self.num_val_imgs = len(self.sample_dataloader)
        val_params = {
            'batch_size': self.params['batch_size'],
            'shuffle': False,
            'drop_last': True,
            'collate_fn': collater,
            'num_workers': self.params['num_workers']}
        self.sample_dataloader = DataLoader(self.sample_dataloader, **val_params)

        return self.sample_dataloader

    

class VAEXperiment(pl.LightningModule):

    def __init__(self,
                 vae_model: BaseVAE,
                 params: dict) -> None:
        super(VAEXperiment, self).__init__()

        self.model = vae_model
        self.params = params
        self.curr_device = None
        self.hold_graph = False
        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx = 0):
        real_img, labels = batch
        self.curr_device = real_img.device
        results = self.forward(real_img, labels = labels)
        train_loss = self.model.loss_function(*results, M_N = self.params['batch_size']/ self.num_train_imgs, optimizer_idx=optimizer_idx, batch_idx = batch_idx)
        self.logger.experiment.log({key: val.item() for key, val in train_loss.items()})
        return train_loss

    def validation_step(self, batch, batch_idx, optimizer_idx = 0):
        real_img, labels = batch
        self.curr_device = real_img.device
        results = self.forward(real_img, labels = labels)
        val_loss = self.model.loss_function(*results,  M_N = self.params['batch_size']/ self.num_val_imgs, optimizer_idx = optimizer_idx, batch_idx = batch_idx)
        return val_loss

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        tensorboard_logs = {'avg_val_loss': avg_loss}
        self.sample_images()
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def sample_images(self):
        # Get sample reconstruction image
        test_input, test_label = next(iter(self.sample_dataloader))
        test_input = test_input.to(self.curr_device)
        test_label = test_label.to(self.curr_device)
        recons = self.model.generate(test_input, labels = test_label)
        vutils.save_image(recons.data,
                          f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/"
                          f"recons_{self.logger.name}_{self.current_epoch}.png",
                          normalize=True,
                          nrow=12)

        # vutils.save_image(test_input.data,
        #                   f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/"
        #                   f"real_img_{self.logger.name}_{self.current_epoch}.png",
        #                   normalize=True,
        #                   nrow=12)

        try:
            samples = self.model.sample(144,
                                        self.curr_device,
                                        labels = test_label)
            vutils.save_image(samples.cpu().data,
                              f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/"
                              f"{self.logger.name}_{self.current_epoch}.png",
                              normalize=True,
                              nrow=12)
        except:
            pass


        del test_input, recons #, samples


    def configure_optimizers(self):

        optims = []
        scheds = []
        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.params['LR'],
                               weight_decay=self.params['weight_decay'])
        optims.append(optimizer)
        # Check if more than 1 optimizer is required (Used for adversarial training)
        try:
            if self.params['LR_2'] is not None:
                optimizer2 = optim.Adam(getattr(self.model,self.params['submodel']).parameters(),
                                        lr=self.params['LR_2'])
                optims.append(optimizer2)
        except:
            pass

        try:
            if self.params['scheduler_gamma'] is not None:
                scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
                                                             gamma = self.params['scheduler_gamma'])
                scheds.append(scheduler)

                # Check if another scheduler is required for the second optimizer
                try:
                    if self.params['scheduler_gamma_2'] is not None:
                        scheduler2 = optim.lr_scheduler.ExponentialLR(optims[1],
                                                                      gamma = self.params['scheduler_gamma_2'])
                        scheds.append(scheduler2)
                except:
                    pass
                return optims, scheds
        except:
            return optims

    @data_loader
    def train_dataloader(self):
        transform = self.data_transforms()
        dataset = VAE_Dataset_folder(self.params['train_data_path'], self.params['train_data_path'])
            # raise ValueError('Undefined dataset type')
        self.num_train_imgs = len(dataset)
        return DataLoader(dataset,
                          batch_size= self.params['batch_size'],
                          shuffle = True,
                          drop_last=True)

    @data_loader
    def val_dataloader(self):
        transform = self.data_transforms()
        self.sample_dataloader = VAE_Dataset_folder(self.params['train_data_path'], self.params['train_data_path'])
        self.num_val_imgs = len(self.sample_dataloader)
        self.sample_dataloader = DataLoader(self.sample_dataloader, batch_size= self.params['batch_size'],
                                            shuffle = True,
                                            drop_last=True)

        return self.sample_dataloader

    def data_transforms(self):
        SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
        SetScale = transforms.Lambda(lambda X: X/X.sum(0).expand_as(X))
        transform =transforms.Compose([
            transforms.Resize(self.params['img_size']),
            # torchvision.transforms.CenterCrop(96),
            # torchvision.transforms.RandomRotation((-10,10)),
            # torchvision.transforms.RandomHorizontalFlip(),
            # torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.ToTensor(),
            # torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        #raise ValueError('Undefined dataset type')
        return transform

class UnetExperiment(pl.LightningModule):

    def __init__(self, 
                model: BaseUnet,
                params: dict) -> None:
        super(UnetExperiment, self).__init__()

        self.model = model
        self.params = params
        self.curr_device = None
        self.hold_graph = False
        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass

    def forward(self, input):
        return self.model(input)
    
    def training_step(self, batch, batch_idx, optimizer_idx = 0):
        real_img, label = bath
        self.curr_device = real_img.device
        mask = self.forward(real_img)
        train_loss = self.model.loss_function(mask, label)
        self.logger.experiment.log({key: val.item() for key, val in train_loss.items()})
        return train_loss

    def validation_step(self, batch, batch_idx, optimizer_idx = 0):
        real_img, label = bath
        self.curr_device = real_img.device
        mask = self.forward(real_img)
        val_loss = self.model.loss_function(mask, label)
        self.logger.experiment.log({key: val.item() for key, val in val_loss.items()})
        return val_loss

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        tensorboard_logs = {'avg_val_loss': avg_loss}
        # self.model.detect_image(img_path)
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

 
    def configure_optimizers(self):

        optims = []
        scheds = []
        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.params['LR'])
        optims.append(optimizer)
        # Check if more than 1 optimizer is required (Used for adversarial training)
        try:
            if self.params['LR_2'] is not None:
                optimizer2 = optim.Adam(getattr(self.model,self.params['submodel']).parameters(),
                                        lr=self.params['LR_2'])
                optims.append(optimizer2)
        except:
            pass

        try:
            if self.params['scheduler_gamma'] is not None:
                scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
                                                             gamma = self.params['scheduler_gamma'])
                scheds.append(scheduler)

                # Check if another scheduler is required for the second optimizer
                try:
                    if self.params['scheduler_gamma_2'] is not None:
                        scheduler2 = optim.lr_scheduler.ExponentialLR(optims[1],
                                                                      gamma = self.params['scheduler_gamma_2'])
                        scheds.append(scheduler2)
                except:
                    pass
                return optims, scheds
        except:
            return optims



   @data_loader
    def train_dataloader(self):
        transform = self.data_transforms()
        dataset = VAE_Dataset_folder(self.params['train_data_path'], self.params['train_data_path'])
            # raise ValueError('Undefined dataset type')
        self.num_train_imgs = len(dataset)
        return DataLoader(dataset,
                          batch_size= self.params['batch_size'], num_workers=self.params['num_workers'],
                          shuffle = True,
                          drop_last=True)

    @data_loader
    def val_dataloader(self):
        transform = self.data_transforms()
        self.sample_dataloader = VAE_Dataset_folder(self.params['train_data_path'], self.params['train_data_path'])
        self.num_val_imgs = len(self.sample_dataloader)
        self.sample_dataloader = DataLoader(self.sample_dataloader, batch_size= self.params['batch_size'],
                                            shuffle = False, num_workers=self.params['num_workers'],
                                            drop_last=True)

        return self.sample_dataloader