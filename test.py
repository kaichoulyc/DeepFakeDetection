import os

import fire
import yaml
import torch

from pytorch_lightning import Trainer
from main_module import get_module
from pytorch_lightning.logging import TestTubeLogger


def make_dir_safe(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def build_trainer(gpus: list,
                  distributed_backend,
                  cp_saver_params: dict,
                  loggs_save_dir: str,
                  experement_name: str):
    cp_save_dir     = cp_saver_params['savedir']
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
    cp_save_dir = os.path.join(cp_save_dir, f'version_{version}')
    make_dir_safe(cp_save_dir)

    tt_logger = TestTubeLogger(save_dir=loggs_save_dir, name=experement_name)
    trainer = Trainer(gpus=gpus,
                      max_nb_epochs=0,
                      distributed_backend=distributed_backend,
                      logger=tt_logger,
                      train_percent_check=0,
                      val_percent_check=0)

    return trainer


def main(config: str):

    with open(config) as f:
        cfg = yaml.safe_load(f)

    experement_name     = cfg['experement_name']
    model_info          = cfg['model_info']
    loss                = cfg['loss']
    gpus                = cfg['gpus']
    base_weights_path   = cfg.get('base_weights')
    cp_saver_params     = cfg['cp_saver_params']
    loggs_save_dir      = cfg['base_logdir']
    opt_name            = cfg['opt_name']
    opt_params          = cfg['opt_params']
    sched_name          = cfg.get('sched_name')
    sched_params        = cfg.get('sched_params')
    loader_data         = cfg['loader_data']
    weights             = cfg['infer_chekpoint']

    distributed_backend = None
    if len(gpus) > 1:
        distributed_backend = 'ddp'

    make_dir_safe(loggs_save_dir)

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
    
    model.load_state_dict(torch.load(weights, map_location='cpu')['state_dict'])
    model.freeze()
    trainer = build_trainer(
        gpus=gpus,
        distributed_backend=distributed_backend,
        cp_saver_params=cp_saver_params,
        loggs_save_dir=loggs_save_dir,
        experement_name=experement_name,
    )

    trainer.test(model)


if __name__ == '__main__':
    fire.Fire(main)