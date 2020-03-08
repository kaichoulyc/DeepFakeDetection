import argparse

import pytorch_lightning as pl
import torch
import torch.nn as nn

from models.model import get_model
from util.data_loader import get_dataloader
from util.losses import FocalLoss
from util.utils import accuracy


class FakeClassificationModule(pl.LightningModule):

    criterrions = {
        'CrossEntropy': nn.CrossEntropyLoss,
        'Focal': FocalLoss
    }

    def __init__(self, hparams):
        super(FakeClassificationModule, self).__init__()

        self.hparams = hparams
        model_info = self.hparams.model_info
        self.model = get_model(model_info['model_type'],
                               model_info['num_classes'],
                               model_info['pretrained'])
        if self.hparams.base_weights is not None:
            weights = torch.load(self.hparams.base_weights, map_location='cpu')
            self.model.load_state_dict(weights['model'], strict=False)
        self.optim_fn = torch.optim.__dict__[self.hparams.opt_name]
        self.loader_data = self.hparams.loader_data
        self.criterrion = self.criterrions[self.hparams.criterion]
        self.ddp = True if len(self.hparams.gpus) > 1 else False

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inp_img, target = batch
        preds = self.forward(inp_img)
        loss = self.criterrion(preds, target)
        tensorboard_logs = {'train_loss': loss}
        results = {'loss': loss, 'log': tensorboard_logs}

        return results

    def validation_step(self, batch, batch_idx):
        inp_img, target = batch
        preds = self.forward(inp_img)
        loss = self.criterrion(preds, target)
        accu = accuracy(preds, target)
        results = {'val_loss': loss,
                   'val_accuracy': accu}

        return results

    def validation_end(self, outputs):

        avg_loss = 0
        avg_accu = 0
        for output in outputs:
            avg_loss += output['val_loss']
            avg_accu += output['val_accuracy']
        avg_loss /= len(outputs)
        avg_accu /= len(outputs)
        tensorboard_logs = {'val_loss': avg_loss,
                            'val_accuracy': avg_accu}
        return {
            'avg_val_loss': avg_loss,
            'log': tensorboard_logs,
            'progress_bar': {'val_loss': avg_loss},
        }

    def test_step(self, batch, batch_idx):
        inp_img, target = batch
        preds = self.forward(inp_img)
        loss = self.criterrion(preds, target)
        accu = accuracy(preds, target)
        results = {'test_loss': loss,
                   'test_accuracy': accu}

        return results

    def test_end(self, outputs):

        avg_loss = 0
        avg_accu = 0
        for output in outputs:
            avg_loss += output['test_loss']
            avg_accu += output['test_accuracy']
        avg_loss /= len(outputs)
        avg_accu /= len(outputs)
        tensorboard_logs = {'test_loss': avg_loss,
                            'test_accuracy': avg_accu}
        return {
            'avg_val_loss': avg_loss,
            'log': tensorboard_logs,
            'progress_bar': {'val_loss': avg_loss},
        }

    def configure_optimizers(self):

        optimizer = self.optim_fn(params=self.parameters(), **self.hparams.opt_params)

        return optimizer

    def train_dataloader(self):
        return get_dataloader(dataset=self.loader_data['dataset'],
                              path=self.loader_data['train_path'],
                              mode='train',
                              side_size=self.loader_data['side_size'],
                              batch_size=self.loader_data['batch_size'],
                              num_workers=self.loader_data['num_workers'],
                              ddp=self.ddp,
                              df_path=self.loader_data.get('df_train_path'))

    def val_dataloader(self):
        return get_dataloader(dataset=self.loader_data['dataset'],
                              path=self.loader_data['valid_path'],
                              mode='train',
                              side_size=self.loader_data['side_size'],
                              batch_size=self.loader_data['batch_size'],
                              num_workers=self.loader_data['num_workers'],
                              ddp=self.ddp,
                              df_path=self.loader_data.get('df_valid_path'))

    def test_dataloader(self):
        return get_dataloader(dataset=self.loader_data['dataset'],
                              path=self.loader_data['test_path'],
                              mode='test',
                              side_size=self.loader_data['side_size'],
                              batch_size=self.loader_data['batch_size'],
                              num_workers=self.loader_data['num_workers'],
                              ddp=self.ddp,
                              df_path=self.loader_data.get('df_test_path'))


def get_module(
    exp_name: str,
    model_info: dict,
    criterion: str,
    opt_name: str,
    opt_params: dict,
    loader_data: dict,
    sched_name: str,
    sched_params: dict,
    gpus: str,
    base_weights
):

    hparams = dict(
        exp_name=exp_name,
        model_info=model_info,
        criterion=criterion,
        opt_name=opt_name,
        opt_params=opt_params,
        loader_data=loader_data,
        sched_name=sched_name,
        sched_params=sched_params,
        gpus=gpus,
        base_weights=base_weights
    )
    hparams = argparse.Namespace(**hparams)
    module = FakeClassificationModule(hparams)

    return module
