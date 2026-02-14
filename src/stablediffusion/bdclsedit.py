import torch
from tqdm import tqdm
from src.stablediffusion.utils import encode_text, sample_its_from_i0, get_variance
from src.stablediffusion.ctfds import CTFDSLoss
from torch.optim.sgd import SGD
from src.stablediffusion.utils import reverse_step, reverse_step_from_it, initialize_theta_t


def u_inference(
    model, 
    i0,
    text_embeddings, uncond_embedding, timesteps,
    prog_bar=False,
    cfg_scale=3.5,
    num_inference_steps=50, eps=None,
    get_eps=False):
    """
    Adapted from DDPM_inversion (MIT License):
    https://github.com/inbarhub/DDPM_inversion
    """
    
    latent_height, latent_width = i0.shape[2], i0.shape[3]
    variance_noise_shape = (
        num_inference_steps,
        model.unet.config.in_channels,
        latent_height,
        latent_width)

    its, _ = sample_its_from_i0(model, i0, num_inference_steps=num_inference_steps)
    alpha_bar = model.scheduler.alphas_cumprod
    us = torch.zeros(size=variance_noise_shape, device=model.device)

    t_to_idx = {int(v): k for k, v in enumerate(timesteps)}
    it = i0
    # op = tqdm(reversed(timesteps)) if prog_bar else reversed(timesteps)
    op = tqdm(timesteps) if prog_bar else timesteps

    for t in op:
        idx = num_inference_steps - t_to_idx[int(t)] - 1

        it = its[idx + 1][None]

        with torch.no_grad():
            out = model.unet.forward(it, timestep=t, encoder_hidden_states=uncond_embedding,)
            cond_out = model.unet.forward(it, timestep=t, encoder_hidden_states=text_embeddings)

        noise_pred = out.sample + cfg_scale * (cond_out.sample - out.sample)

        itm1 = its[idx][None]

        pred_original_sample = (it - (1 - alpha_bar[t]) ** 0.5 * noise_pred) / alpha_bar[t] ** 0.5

        prev_timestep = t - model.scheduler.config.num_train_timesteps // model.scheduler.num_inference_steps
        alpha_prod_t_prev = model.scheduler.alphas_cumprod[
            prev_timestep] if prev_timestep >= 0 else model.scheduler.final_alpha_cumprod

        variance = get_variance(model, t)
        pred_sample_direction = (1 - alpha_prod_t_prev -  variance) ** (0.5) * noise_pred

        mu_it = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction

        u = (itm1 - mu_it) / (variance ** 0.5)
        us[idx] = u

        itm1 = mu_it + (variance ** 0.5) * u
        its[idx] = itm1

    us[0] = torch.zeros_like(us[0])

    return us, its


def bdclsedit(
    pipe, 
    source_prompt, 
    target_prompt, source_image, 
    cfg_scale=3.5, num_inference_steps=50, 
    skip=0.4, opt_iter=10,
    device="cuda", prog_bar=True,
    clip_min=-0.2, clip_max=1.2,
    theta_expand=False,
    lamda=0.1,
    theta_init_threshold=0.7,
    theta_init_low=0.2, # if idx > threshold else 1.0
    n_select_ratio=0.2,
    lr=0.1,
):

    timesteps = pipe.scheduler.timesteps.to(pipe.device)
    with torch.no_grad():
        i_source = pipe.vae.encode(source_image)['latent_dist'].mean * 0.18215

    source_embedding = encode_text(pipe, source_prompt)
    target_embedding = encode_text(pipe, target_prompt)
    uncond_embedding = encode_text(pipe, "")

    us, its = u_inference(pipe, i_source, source_embedding, uncond_embedding, timesteps, 
                          cfg_scale=cfg_scale, num_inference_steps=num_inference_steps, prog_bar=prog_bar)

    ctfds_loss = CTFDSLoss(pipe.device, pipe, dtype=torch.float32)

    op = tqdm(timesteps)

    if theta_expand:
        thetas = [torch.zeros(target_embedding.shape, device=pipe.device) for _ in range(num_inference_steps)]
    else:
        thetas = torch.zeros(num_inference_steps).to(pipe.device)

    t_to_idx = {int(v): k for k, v in enumerate(timesteps)}

    n_select = max(1, int(len(timesteps) * n_select_ratio))
    perm = torch.randperm(len(timesteps), device=pipe.device)
    selected_timesteps = timesteps[perm[:n_select]]

    it = its[int(num_inference_steps * (1 - skip))].expand(1, -1, -1, -1)

    for t in op:
        idx = num_inference_steps - t_to_idx[int(t)] - 1
        if idx < num_inference_steps * (1 - skip):
            with torch.no_grad():
                uncond_out = pipe.unet.forward(it, timestep=t,
                                            encoder_hidden_states=uncond_embedding)
            ut = us[idx].expand(1, -1, -1, -1)
            if idx > 1:
                expand_shape = target_embedding.shape if theta_expand else None
                theta_t_raw = initialize_theta_t(pipe, idx, expand_to_shape=expand_shape, 
                theta_threshold=theta_init_threshold, theta_init_low=theta_init_low)
                optimizer = SGD(params=[theta_t_raw], lr=lr)
                with torch.no_grad():
                    if t in selected_timesteps:
                        it_source_prev = its[idx-1].unsqueeze(0)
                        it_target_prev = reverse_step_from_it(
                            pipe, it, t, cond_embedding=target_embedding, 
                            uncond_out=uncond_out, 
                            variance_noise=ut, 
                            cfg_scale=cfg_scale,
                            )
                        it_tilde_prev = ctfds_loss.ctf_reference(
                            it_source_prev, it_target_prev, 
                            target_embedding, 
                            source_embedding, 
                            uncond_embedding, 
                            timestep=t,
                            lamda=lamda,
                            guidance_scale=cfg_scale,
                            )
                    else:
                        it_tilde_prev = it

                for _ in range(opt_iter):
                    theta_t = theta_t_raw
                    mix_embedding = target_embedding.detach() * theta_t + source_embedding.detach() * (1 - theta_t)
                    it_ctf_prev = reverse_step_from_it(
                        pipe, it, t, cond_embedding=mix_embedding, 
                        uncond_out=uncond_out, 
                        variance_noise=ut, 
                        cfg_scale=cfg_scale,
                        )

                    loss = ctfds_loss.get_ctf_loss(
                        it_tilde_prev, 
                        it_ctf_prev, 
                        mix_embedding, 
                        source_embedding, 
                        uncond_embedding, 
                        timestep=t,
                        guidance_scale=cfg_scale,
                    )
                    if theta_expand:
                        scale = theta_t_raw.numel() ** 0.5
                    else:
                        scale = 1.0
                    loss_scale = 2000 * scale

                    optimizer.zero_grad()
                    (loss_scale * loss).backward(retain_graph=True)
                    optimizer.step()

            with torch.inference_mode():
                if idx > 1:
                    theta_t = torch.clamp(theta_t_raw.detach(), min=clip_min, max=clip_max)
                else:
                    if theta_expand:
                        theta_t = torch.zeros(target_embedding.shape, device=pipe.device)
                    else:
                        theta_t = torch.tensor([0], device=pipe.device)
                mix_embedding = target_embedding.detach() * theta_t + source_embedding.detach() * (1 - theta_t)
                
                it_target = reverse_step_from_it(
                    pipe, it, t, cond_embedding=mix_embedding, 
                    uncond_out=uncond_out, 
                    variance_noise=ut, 
                    cfg_scale=cfg_scale,
                    )
                if theta_expand:
                    thetas[idx] = theta_t.detach().clone()
                else:
                    thetas[idx] = theta_t.squeeze()
            it = it_target.clone()
            del it_target

    return it
