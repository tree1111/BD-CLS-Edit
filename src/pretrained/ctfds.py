from typing import Tuple, Union, Optional, List

import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel

from src.pretrained.utils import stack_embeddings
from src.pretrained.utils import reverse_step

def init_pipe(device, dtype, unet, scheduler) -> Tuple[UNet2DConditionModel, torch.Tensor, torch.Tensor]:
    with torch.inference_mode():
        alphas = torch.sqrt(scheduler.alphas_cumprod).to(device, dtype=dtype)
        sigmas = torch.sqrt(1 - scheduler.alphas_cumprod).to(device, dtype=dtype)
    for p in unet.parameters():
        p.requires_grad = False
    return unet, alphas, sigmas


class CTFDSLoss:

    def noise_input(self, z, eps=None, timestep=None):
        if timestep is None:
            b = z.shape[0]
            timestep = torch.randint(
                low=self.t_min,
                high=min(self.t_max, 1000) - 1,  # Avoid the highest timestep.
                size=(b,),
                device=z.device, dtype=torch.long)

        if eps is None:
            eps = torch.randn_like(z)
        alpha_t = self.alphas[timestep, None, None, None]
        sigma_t = self.sigmas[timestep, None, None, None]
        z_t = alpha_t * z + sigma_t * eps
        return z_t, eps, timestep, alpha_t, sigma_t

    def get_eps_prediction(
        self, z_t: torch.Tensor, 
        timestep: torch.Tensor, 
        text_embeddings: torch.Tensor, 
        alpha_t: torch.Tensor, 
        sigma_t: torch.Tensor, 
        get_raw=False,
        guidance_scale=7.5
        ):

        latent_input = torch.cat([z_t] * 2)
        timestep = torch.cat([timestep] * 2)
        embedd = text_embeddings.permute(1, 0, 2, 3).reshape(-1, *text_embeddings.shape[2:])
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            e_t = self.unet(latent_input, timestep, embedd).sample
            if self.prediction_type == 'v_prediction':
                e_t = torch.cat([alpha_t] * 2) * e_t + torch.cat([sigma_t] * 2) * latent_input
            e_t_uncond, e_t = e_t.chunk(2)
            if get_raw:
                return e_t_uncond, e_t
            e_t = e_t_uncond + guidance_scale * (e_t - e_t_uncond)
            assert torch.isfinite(e_t).all()
        if get_raw:
            return e_t
        pred_z0 = (z_t - sigma_t * e_t) / alpha_t
        return e_t, pred_z0

    def ctf_reference(
        self,  
        it_source_prev: torch.Tensor, 
        it_target_prev: torch.Tensor, 
        target_embedding: torch.Tensor, 
        source_embedding: torch.Tensor,
        uncond_embedding: torch.Tensor,
        timestep: Optional[int] = None,
        lamda: float = 0.1,
        guidance_scale=7.5,
        use_sigma_sq_grad: bool = True) -> torch.Tensor:

        timestep = timestep.to(torch.long).view(1)
        source_embedding_stack, target_embedding_stack = stack_embeddings(target_embedding, source_embedding, uncond_embedding)

        prev_timestep = (timestep - self.inference_steps_delta).clamp(min=0)
        alpha_t = self.alphas[prev_timestep, None, None, None]
        sigma_t = self.sigmas[prev_timestep, None, None, None]

        eps_pred, _ = self.get_eps_prediction(torch.cat((it_source_prev, it_target_prev)),
                                        torch.cat((prev_timestep, prev_timestep)),
                                        torch.cat((source_embedding_stack, target_embedding_stack)),
                                        torch.cat((alpha_t, alpha_t)),
                                        torch.cat((sigma_t, sigma_t)),
                                        guidance_scale=guidance_scale,
                                        )
        eps_pred_source, eps_pred_target = eps_pred.chunk(2)
        if use_sigma_sq_grad:
            grad = (sigma_t ** 2) * (eps_pred_target - eps_pred_source)
        else:
            grad = eps_pred_target - eps_pred_source

        it_tilde_prev = it_source_prev + grad.clone() * lamda
        return it_tilde_prev

    def get_ctf_loss(
        self, 
        it_tilde_prev: torch.Tensor, 
        it_ctf_prev: torch.Tensor, 
        mixed_embedding: torch.Tensor, 
        source_embedding: torch.Tensor, 
        uncond_embedding: torch.Tensor,
        timestep: Optional[int] = None,
        guidance_scale=7.5,
        symmetric=False,
        reduction='mean',
        use_sigma_exp_grad: bool = True,
        ) -> torch.Tensor:

        source_embedding_stack, target_embedding_stack = stack_embeddings(mixed_embedding, source_embedding, uncond_embedding)

        timestep = timestep.to(torch.long).view(1)
        prev_timestep = (timestep - self.inference_steps_delta).clamp(min=0)

        alpha_t = self.alphas[prev_timestep, None, None, None]
        sigma_t = self.sigmas[prev_timestep, None, None, None]
        with torch.inference_mode():
            eps_pred, _ = self.get_eps_prediction(torch.cat((it_tilde_prev, it_ctf_prev)),
                                                  torch.cat((timestep, timestep)),
                                                  torch.cat((source_embedding_stack, target_embedding_stack)),
                                                  torch.cat((alpha_t, alpha_t)),
                                                  torch.cat((sigma_t, sigma_t)),
                                                  guidance_scale=guidance_scale)
            eps_pred_source, eps_pred_target = eps_pred.chunk(2)
            if use_sigma_exp_grad:
                grad = (sigma_t ** self.sigma_exp) * (eps_pred_target - eps_pred_source)
            else:
                grad = eps_pred_target - eps_pred_source
        loss = it_ctf_prev * grad.clone()

        if symmetric:
            loss = loss.sum() / (it_ctf_prev.shape[2] * it_ctf_prev.shape[3])
            loss_symm = self.rescale * it_tilde_prev * (-grad.clone())
            loss += loss_symm.sum() / (it_ctf_prev.shape[2] * it_ctf_prev.shape[3])
        elif reduction == 'mean':
            loss = loss.sum() / (it_ctf_prev.shape[2] * it_ctf_prev.shape[3])

        return loss

    def __init__(self, device, pipe: StableDiffusionPipeline, dtype=torch.float32):
        self.t_min = 50
        self.t_max = 950
        self.alpha_exp = 0
        self.sigma_exp = 0
        self.dtype = dtype
        self.unet, self.alphas, self.sigmas = init_pipe(device, dtype, pipe.unet, pipe.scheduler)
        self.prediction_type = pipe.scheduler.config.prediction_type
        self.inference_steps_delta = pipe.scheduler.config.num_train_timesteps // pipe.scheduler.num_inference_steps


