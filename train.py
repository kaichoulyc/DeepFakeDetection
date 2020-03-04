import os

import fire
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.logging import TestTubeLogger

from main_module import get_module


def make_dir_safe(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def build_trainer(experement_name: str,
                  gpus: list,
                  epochs: int,
                  use_fp16: bool,
                  distributed_backend,
                  train_percent_check: float,
                  val_percent_check: float,
                  early_stop_params: dict,
                  cp_saver_params: dict,
                  loggs_save_dir: str,
                  gradient_clip_val: float,
                  resume_checkpoint_path: str):

    patience        = early_stop_params['patience']
    cp_save_dir     = cp_saver_params['savedir']
    cp_metric       = cp_saver_params['metric']
    cp_mode         = cp_saver_params['mode']
    cp_prefix       = cp_saver_params['prefix']

    cp_save_dir = os.path.join(cp_save_dir, experement_name)
    make_dir_safe(cp_save_dir)
    version_check = os.path.join(loggs_save_dir, experement_name)
    make_dir_safe(version_check)
    versions = os.listdir(version_check)
    version = 0
    if versions:
        numbers = [int(i.split('_')[1]) for i in versions]
        numbers.sort()
        version = numbers[-1]
        version += 1
    print(f'Version: {version}')
    cp_save_dir = os.path.join(cp_save_dir, f'version_{version}')
    make_dir_safe(cp_save_dir)

    early_stop_callback = EarlyStopping(patience=patience)
    checkpoint_callback = ModelCheckpoint(
        filepath=cp_save_dir,
        monitor=cp_metric,
        mode=cp_mode,
        prefix=cp_prefix
    )
    tt_logger = TestTubeLogger(save_dir=loggs_save_dir, name=experement_name)
    trainer = Trainer(gpus=gpus,
                      max_nb_epochs=epochs,
                      early_stop_callback=early_stop_callback,
                      distributed_backend=distributed_backend,
                      checkpoint_callback=checkpoint_callback,
                      logger=tt_logger,
                      train_percent_check=train_percent_check,
                      val_percent_check=val_percent_check,
                      use_amp=use_fp16,
                      gradient_clip_val=gradient_clip_val,
                      resume_from_checkpoint=resume_checkpoint_path)

    return trainer


def main(config: str = "config.yml", local_rank: int = 0):

    with open(config) as f:
        cfg = yaml.safe_load(f)

    experement_name     = cfg['experement_name']
    model_info          = cfg['model_info']
    loss                = cfg['loss']
    gpus                = cfg['gpus']
    use_fp16            = cfg['use_fp16']
    base_weights_path   = cfg.get('base_weights')
    train_percent_check = cfg['train_percent_check']
    val_percent_check   = cfg['val_percent_check']
    early_stop_params   = cfg['early_stop_params']
    cp_saver_params     = cfg['cp_saver_params']
    loggs_save_dir      = cfg['base_logdir']
    gradient_clip_val   = cfg.get('gradient_clip_val')
    epochs              = cfg['epochs']
    opt_name            = cfg['opt_name']
    opt_params          = cfg['opt_params']
    sched_name          = cfg.get('sched_name')
    sched_params        = cfg.get('sched_params')
    loader_data         = cfg['loader_data']

    if len(gpus) > 1:
        distributed_backend = 'ddp'

    make_dir_safe(loggs_save_dir)
    resume = cfg.get('resume')
    resume_checkpoint_path = None
    if resume is not None:
        resume_checkpoint_path = resume['checkpoint_path']

    trainer = build_trainer(experement_name=experement_name,
                            gpus=gpus,
                            epochs=epochs,
                            use_fp16=use_fp16,
                            distributed_backend=distributed_backend,
                            train_percent_check=train_percent_check,
                            val_percent_check=val_percent_check,
                            early_stop_params=early_stop_params,
                            cp_saver_params=cp_saver_params,
                            loggs_save_dir=loggs_save_dir,
                            gradient_clip_val=gradient_clip_val,
                            resume_checkpoint_path=resume_checkpoint_path)

    model = get_module(exp_name=experement_name,
                       model_info=model_info,
                       criterion=loss,
                       opt_name=opt_name,
                       opt_params=opt_params,
                       loader_data=loader_data,
                       sched_name=sched_name,
                       sched_params=sched_params,
                       gpus=gpus,
                       base_weights=base_weights_path)

    trainer.fit(model)


if __name__ == '__main__':
    fire.Fire(main)
    
