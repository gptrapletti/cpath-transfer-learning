import os
import yaml
import pytorch_lightning as pl
from src.datamodule import CONICDataModule
from src.model import SegResNet

with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)
    
datamodule = CONICDataModule(data_dir=cfg['data_dir'], biomarker=2)

model = SegResNet(
    ckpt_path=cfg['pretrained_model_path'],
    pruning = 3,
    lr = cfg['lr']
)

callbacks = [
    pl.callbacks.ModelCheckpoint(
        dirpath = os.path.join('checkpoints', cfg['run_name']),
        filename = '{epoch}',
        monitor = 'val_loss',
        mode = 'min',
        save_top_k = 1    
    ),
    pl.callbacks.LearningRateMonitor(logging_interval='epoch') 
]

logger = pl.loggers.MLFlowLogger(
    experiment_name = cfg['experiment_name'],
    run_name = cfg['run_name'],
)

trainer = pl.Trainer(
    max_epochs=cfg['epochs'],
    accelerator='gpu',
    logger=logger,
    callbacks=callbacks
)

trainer.fit(model=model, datamodule=datamodule)