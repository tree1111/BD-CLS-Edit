import os
import glob
import shutil
import hashlib
import json

import torch
import numpy as np
import pytorch_lightning as pl

from src.ds.causal_graph import CausalGraph

from src.colormnistbar.gen.pipeline import Pipeline


def create_trainer(directory, max_epochs, gpu=None):
    checkpoint = pl.callbacks.ModelCheckpoint(dirpath=f'{directory}/checkpoints/', monitor="train_loss", save_top_k=3)
    return pl.Trainer(
        callbacks=[
            checkpoint
        ],
        max_epochs=max_epochs,
        accumulate_grad_batches=1,
        logger=pl.loggers.TensorBoardLogger(f'{directory}/logs/'),
        log_every_n_steps=1,
        terminate_on_nan=True,
        gpus=gpu
    ), checkpoint


def gen_runner(name, cg_file, hyperparams, gpu, seed=42):
    d = "out/%s/%s/gen" % (name, hyperparams['graph'])

    cg = CausalGraph.read(cg_file)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    if os.path.isfile(f"{d}/saved_models/best.ckpt"):
        if gpu is not None:
            map_device = torch.device('cuda:' + str(gpu[0]))
        else:
            map_device = torch.device('cpu')
        print('gen training [done], loading model from ', d)
        model = Pipeline(cg=cg, hyperparams=hyperparams)
        checkpoint = torch.load(f"{d}/saved_models/best.ckpt", map_location=map_device)
        model.load_state_dict(checkpoint['state_dict'])
        model.to(map_device)
        return model

    if hyperparams is None:
        hyperparams = dict(h_layers=2, h_size=64, u_size=4)

    model = Pipeline(cg=cg, hyperparams=hyperparams)
    trainer, checkpoint = create_trainer(d, hyperparams['max-epoch-gen'], gpu)
    trainer.fit(model)
    
    ckpt = torch.load(checkpoint.best_model_path)
    model.load_state_dict(ckpt['state_dict'])

    if not os.path.isdir(f"{d}/saved_models"):
        os.mkdir(f"{d}/saved_models")
    trainer.save_checkpoint(
        f"{d}/saved_models/best.ckpt")
    return model