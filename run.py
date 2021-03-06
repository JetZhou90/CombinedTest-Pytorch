import yaml
import argparse
import numpy as np
import torch
from models import vae_models, unet_models, DetectModel
from experiment import VAEXperiment, EfficientExperiment, UnetExperiment
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.logging import TestTubeLogger


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generic runner for seal-validation system models')
    parser.add_argument('--config',  '-c',
                        dest="filename",
                        metavar='FILE',
                        help =  'path to the config file',
                        default='configs/unet.yaml')
    parser.add_argument('--type', '-t',
                        dest="model_type",
                        help =  'choose the model type- 0-detect model, 1-segment model, 2-generative model',
                        type= int,
                        default=1)
    args = parser.parse_args()
    with open(args.filename, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    tt_logger = TestTubeLogger(
        save_dir=config['logging_params']['save_dir'],
        name=config['logging_params']['name'],
        debug=False,
        create_git_tag=False,
    )
    # For reproducibility
    torch.manual_seed(config['logging_params']['manual_seed'])
    np.random.seed(config['logging_params']['manual_seed'])
    cudnn.deterministic = True
    cudnn.benchmark = False
    experiment_type = int(args.model_type)
    if experiment_type == 0:
        model = DetectModel(num_classes=config['model_params']['num_classes'], 
                        compound_coef=config['model_params']['compound_coef'], 
                        obj_list=config['exp_params']['obj_list'], 
                        ratios=eval(config['model_params']['anchors_ratios']), 
                        scales=eval(config['model_params']['anchors_scales']))
        experiment = EfficientExperiment(model, config['exp_params'])    
    elif experiment_type == 1:
        model = unet_models[config['model_params']['name']]()
        experiment = UnetExperiment(model, config['exp_params'])
    elif experiment_type == 2:
        model = vae_models[config['model_params']['name']](**config['model_params'])
        experiment = VAEXperiment(experiment = UnetExperiment(model, config['exp_params']))
    else:
        raise 'Unknow Type number -- Choose the model type- 0-detect model, 1-segment model, 2-generative model'   
    runner = Trainer(default_save_path=f"{tt_logger.save_dir}", min_nb_epochs=1, logger=tt_logger, log_save_interval=100, train_percent_check=1., val_percent_check=1., num_sanity_val_steps=5, early_stop_callback = False, **config['trainer_params'])
    #runner = Trainer(resume_from_checkpoint='./logs/EfficientNet/version_2/checkpoints/_ckpt_epoch_236.ckpt',min_nb_epochs=1, logger=tt_logger, log_save_interval=100, train_percent_check=1., val_percent_check=1., num_sanity_val_steps=5, early_stop_callback = False, **config['trainer_params'])
    print(f"======= Training {config['model_params']['name']} =======")
    runner.fit(experiment)