from src.colormnistbar.fi.ddpm import FiPipline
import torch
import os


def fi_runner(name, hyperparams, filename=None, gpu=None):
    d = "out/%s/%s/fi" % (name, hyperparams['graph'])

    if gpu:
        device = torch.device('cuda:' + str(gpu[0]))
    else:
        device = torch.device('cpu')
    
    if hyperparams['graph'] == "full-ncm":
        c_size = 16
    elif hyperparams['graph'] == "cls-digit":
        c_size = 14
    elif hyperparams['graph'] == "cls-color":
        c_size = 12
    else:
        raise ValueError(f"Graph {hyperparams['graph']} not supported")

    if os.path.isfile(f"{d}/saved_models/best.pth"):
        print('fi training [done], loading model from ', d)
        if filename:
            model = FiPipline(
                device=device,
                save_path=d,
                checkpoint_name=filename, 
                c_size=c_size
                )
        else:
            model = FiPipline(
                device=device,
                save_path=d,
                checkpoint_name="best.pth",
                c_size=c_size
                )
        return model


    model = FiPipline(
        device=device,
        save_path=d,
        c_size=c_size,
        )

    model.train(
        batch_size=hyperparams['batch-size'], 
        n_epoch=hyperparams['max-epoch-fi'], 
        lr=hyperparams['lr-fi'],
        timesteps=hyperparams['timesteps'], 
        beta1=hyperparams['beta1'], beta2=hyperparams['beta2'],
        )

    return model
    