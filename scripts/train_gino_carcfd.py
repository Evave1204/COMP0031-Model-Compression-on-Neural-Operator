import torch
import wandb
import sys
from configmypy import ConfigPipeline, YamlConfig, ArgparseConfig
from neuralop.training import setup, AdamW
from neuralop import get_model
from neuralop.utils import get_wandb_api_key
from neuralop.losses.data_losses import LpLoss
from neuralop.training.trainer import Trainer
from neuralop.data.datasets import CarCFDDataset
from neuralop.data.transforms.data_processors import DataProcessor
from copy import deepcopy
from neuralop.data.transforms.gino_processor import GINOCFDDataProcessor

# query points is [sdf_query_resolution] * 3 (taken from config ahmed)
# Read the configuration
config_name = 'cfd'
pipe = ConfigPipeline([YamlConfig('./gino_carcfd_config.yaml', config_name=config_name, config_folder='./config'),
                       ArgparseConfig(infer_types=True, config_name=None, config_file=None),
                       YamlConfig(config_folder='./config')
                      ])
config = pipe.read_conf()

#Set-up distributed communication, if using
device, is_logger = setup(config)

if config.data.sdf_query_resolution < config.gino.fno_n_modes[0]:
    config.gino.fno_n_modes = [config.data.sdf_query_resolution]*3

#Set up WandB logging
wandb_init_args = {}
config_name = 'car-pressure'
if config.wandb.log and is_logger:
    wandb.login(key=get_wandb_api_key())
    
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M')
    
    if config.wandb.name:
        wandb_name = f"{config.wandb.name}-{timestamp}"
    else:
        wandb_name = '_'.join(
            f'{var}' for var in [config_name, config.data.sdf_query_resolution, timestamp])

    wandb_init_args = dict(config=config, 
                           name=wandb_name, 
                           group=config.wandb.group,
                           project=config.wandb.project,
                           entity=config.wandb.entity)

    if config.wandb.sweep:
        for key in wandb.config.keys():
            config.params[key] = wandb.config[key]
    wandb.init(**wandb_init_args)

#Load CFD body data
data_module = CarCFDDataset(root_dir=config.data.root, 
                             query_res=[config.data.sdf_query_resolution]*3, 
                             n_train=config.data.n_train, 
                             n_test=config.data.n_test, 
                             download=config.data.download
                             )


train_loader = data_module.train_loader(batch_size=1, shuffle=True)
test_loader = data_module.test_loader(batch_size=1, shuffle=False)

model = get_model(config)

#Create the optimizer
optimizer = AdamW(model.parameters(), 
                                lr=config.opt.learning_rate, 
                                weight_decay=config.opt.weight_decay)

if config.opt.scheduler == 'ReduceLROnPlateau':
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=config.opt.gamma, patience=config.opt.scheduler_patience, mode='min')
elif config.opt.scheduler == 'CosineAnnealingLR':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.opt.scheduler_T_max)
elif config.opt.scheduler == 'StepLR':
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                step_size=config.opt.step_size,
                                                gamma=config.opt.gamma)
else:
    raise ValueError(f'Got {config.opt.scheduler=}')


l2loss = LpLoss(d=2,p=2)

if config.opt.training_loss == 'l2':
    train_loss_fn = l2loss
else: 
    raise ValueError(f'Got {config.opt.training_loss=}')

if config.opt.testing_loss == 'l2':
    test_loss_fn = l2loss
else:
    raise ValueError(f'Got {config.opt.testing_loss=}')


output_encoder = deepcopy(data_module.normalizers['press']).to(device)
data_processor = GINOCFDDataProcessor(normalizer=output_encoder, device=device)

trainer = Trainer(model=model, 
                  n_epochs=config.opt.n_epochs,
                  data_processor=data_processor,
                  device=device,
                  wandb_log=config.wandb.log,
                  verbose=is_logger
                  )

if config.wandb.log:
    wandb.log({'time_to_distance': data_module.time_to_distance}, commit=False)

trainer.train(
              train_loader=train_loader,
              test_loaders={'test':test_loader},
              optimizer=optimizer,
              scheduler=scheduler,
              training_loss=train_loss_fn,
              eval_losses={config.opt.testing_loss: test_loss_fn},
              regularizer=None,)

if config.wandb.log and is_logger:
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M')
    model_name = f"model-{config.wandb.name}-{timestamp}"
    
    torch.save(model.state_dict(), f"{model_name}.pt")

    artifact = wandb.Artifact(
        name=model_name,
        type="model",
        description="GINO CarCFD model"
    )
    artifact.add_file(f"{model_name}.pt")
    wandb.log_artifact(artifact)

    wandb.run.log_code(".", include_fn=lambda path: path.endswith(".py") or path.endswith(".yaml"))
    
    wandb.finish()