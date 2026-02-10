
import argparse

from src.colormnistbar.gen.runner import gen_runner

from src.colormnistbar.fi.fi_runner import fi_runner

from src.colormnistbar.ctf_edit_eval import ctf_edit_eval

import torch

parser = argparse.ArgumentParser(description="Basic Runner")

parser.add_argument('--name', default='colormnistbar', help="name of the experiment")

parser.add_argument('--max-epoch-gen', type=int, default=200, help="number of epochs at the generative level")
parser.add_argument('--lr-gen', type=float, default=1e-4, help="optimizer learning rate (default: 1e-4)")
parser.add_argument('--h-layers', type=int, default=2, help="number of hidden layers (default: 2)")
parser.add_argument('--h-size', type=int, default=64, help="mlp hidden layer size (default: 64)")
parser.add_argument('--u-size', type=int, default=4, help="dimensionality of U variables")
parser.add_argument('--gan-mode', default="wgan", help="GAN loss function (default: wgan)")
parser.add_argument('--neural-pu', action="store_true", help="use neural PU distribution")
parser.add_argument('--gen-sigmoid', action="store_true", default=True, help="use sigmoids in generator")
parser.add_argument('--layer-norm', action="store_true", default=True, help="set flag to use layer norm")

parser.add_argument('--max-epoch-fi', type=int, default=100, help="number of epochs at the image level")
parser.add_argument('--lr-fi', type=float, default=1e-3, help="optimizer learning rate (default: 1e-3)")
parser.add_argument('--batch-size', type=int, default=100, help="number of epochs (default: 100)")
parser.add_argument("--timesteps", type=int, default=1000, help="Timesteps for DDPM training")
parser.add_argument("--beta1", type=float, default=0.0001, help="Hyperparameter for DDPM")
parser.add_argument("--beta2", type=float, default=0.02, help="Hyperparameter for DDPM training")

parser.add_argument('--graph', '-G', default="full-ncm", help="name of preset graph over V")
parser.add_argument('--gpu', help="GPU to use")

parser.add_argument('--eval', action="store_true", help="evaluate the model")
parser.add_argument('--eval-n', type=int, default=10000, help="number of samples to evaluate")


args = parser.parse_args()

graph_choice = args.graph.lower()

hyperparams = {
    'graph': graph_choice,
    'h-layers': args.h_layers,
    'h-size': args.h_size,
    'u-size': args.u_size,
    'gan-mode': args.gan_mode,
    'neural-pu': args.neural_pu,
    'gen-sigmoid': args.gen_sigmoid,
    'layer-norm': args.layer_norm,
    'lr-gen': args.lr_gen,
    'lr-fi': args.lr_fi,
    'max-epoch-gen': args.max_epoch_gen,
    'max-epoch-fi': args.max_epoch_fi,
    'timesteps': args.timesteps,
    'beta1': args.beta1,
    'beta2': args.beta2,
    'batch-size': args.batch_size,
    'graph': args.graph,
    'gpu': args.gpu,
}

gpu_used = 0 if args.gpu is None else [int(args.gpu)]
graph = args.graph
cg_file = "dat/cg/{}.cg".format(graph)


parser = argparse.ArgumentParser(description="Basic Runner")
print("NCM running at the generative level...")
m = gen_runner(args.name, cg_file, hyperparams=hyperparams, gpu=gpu_used)
print("Generative level done!")
print("NCM running at the image level...")
fi = fi_runner(args.name, hyperparams, gpu=gpu_used)
print("Image level done!")

if args.eval:
    print("Evaluating the model...")
    m.eval()

    if args.graph == 'full-ncm':
        condition = {
            "D": torch.tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            "C": torch.tensor([1, 0]),
            "BC": torch.tensor([1, 0]),
            "BW": torch.tensor([1, 0]),
        }
        do = {
            "D": torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 1, 0]), 
            }
    elif args.graph == 'cls-digit':
        condition = {
            "X": torch.tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            "B": torch.tensor([1, 0]),
            "Z": torch.tensor([1, 0]),
        }
        do = {
            "X": torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 1, 0]), 
        }
    elif args.graph == 'cls-color':
        condition = {
            "B": torch.tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            "X": torch.tensor([0, 1]),
        }
        do = {
            "X": torch.tensor([1, 0]), 
        }

    ctf_edit_eval(
        m, fi, 
        args.name, 
        args.eval_n, 
        hyperparams['graph'], 
        condition=condition, do=do, 
        hyperparams=hyperparams
        )







