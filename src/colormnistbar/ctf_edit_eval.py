import os
import torch 
from torchvision.utils import save_image

def ctf_edit_eval(m, fi, name, eval_n=100000, graph="full-ncm", condition={}, do={}, hyperparams={}):
    d = "out/%s/%s/eval" % (name, hyperparams['graph'])
    os.makedirs(d, exist_ok=True)
    u = m.ncm.pu.sample(n=eval_n)
    factual_v = m.ncm.forward(u=u, evaluating=True)
    
    if condition:
        mask = torch.ones(eval_n, dtype=torch.bool, device=m.device)
        for k, cond in condition.items():
            cond_t = torch.as_tensor(cond).view(1, -1).to(m.device)
            mask = mask & torch.all(factual_v[k] == cond_t, dim=1)
        idx = torch.where(mask)[0]
        u = {k: v[idx] for k, v in u.items()}
        factual_v = {k: v[idx] for k, v in factual_v.items()}
        n = idx.numel()

        factual_v = m.ncm.forward(u=u, evaluating=True)
    else:
        raise ValueError("Condition not supported")

    if n == 0:
        raise ValueError(
            f"No samples match the condition for eval_n={eval_n}. "
            "Increase eval_n or relax the condition."
        )

    dos = {k: torch.as_tensor(do[k]).view(1, -1).repeat(n, 1) for k in do}
    ctf_v = m.ncm.forward(u=u, do=dos, evaluating=True)

    iT = torch.randn(n, fi.nn_model.in_channels,
                        fi.nn_model.height, fi.nn_model.width,
                        device=fi.device)

    zs = torch.randn(n, hyperparams['timesteps'], fi.nn_model.in_channels,
                        fi.nn_model.height, fi.nn_model.width,
                        device=fi.device)

    if "full" in graph:
        order = ['D', 'C', 'BC', 'BW']
    elif "digit" in graph:
        order = ['X', 'B', 'Z']
    elif "color" in graph:
        order = ['X', 'B']
    else:
        raise ValueError(f"Graph type {graph} not supported")


    factual_v = torch.cat([factual_v[k] for k in order], dim=1)
    ctf_v = torch.cat([ctf_v[k] for k in order], dim=1)

    factual_i, _, _ = fi.sample_ddpm(n, iT=iT, zs=zs, context=factual_v,
                                                                    timesteps=hyperparams['timesteps'],
                                                                    beta1=hyperparams['beta1'], beta2=hyperparams['beta2'],
                                                                    save_rate=20,
                                                                    inference_transform=lambda x: (x + 1) / 2)

    ctf_i, _, _ = fi.sample_ddpm(n, iT=iT, zs=zs, context=ctf_v,
                                                                    timesteps=hyperparams['timesteps'],
                                                                    beta1=hyperparams['beta1'], beta2=hyperparams['beta2'],
                                                                    save_rate=20,
                                                                    inference_transform=lambda x: (x + 1) / 2)

    save_image(factual_i, f"{d}/intial_images.png")
    save_image(ctf_i, f"{d}/edited_images.png")