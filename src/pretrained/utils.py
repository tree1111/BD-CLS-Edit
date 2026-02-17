
import torch


def encode_text(model, prompts, sdxl=True):
    if sdxl:
        tokenizer_one = model.tokenizer  # for OpenCLIP
        tokenizer_two = model.tokenizer_2  # for T5
        text_encoder_one = model.text_encoder
        text_encoder_two = model.text_encoder_2

        text_inputs_one = tokenizer_one(
            prompts, padding="max_length", max_length=tokenizer_one.model_max_length,
            truncation=True, return_tensors="pt"
        )
        text_embeddings_one = text_encoder_one(text_inputs_one.input_ids.to(model.device))[0]

        text_inputs_two = tokenizer_two(
            prompts, padding="max_length", max_length=tokenizer_two.model_max_length,
            truncation=True, return_tensors="pt"
        )
        text_embeddings_two = text_encoder_two(text_inputs_two.input_ids.to(model.device),
                                               output_hidden_states=True).last_hidden_state
        text_encoding = torch.cat([text_embeddings_one, text_embeddings_two], dim=-1)

    else:
        text_input = model.tokenizer(
            prompts,
            padding="max_length",
            max_length=model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            text_encoding = model.text_encoder(text_input.input_ids.to(model.device))[0]
            # text_encoding = model.text_encoder(text_input.input_ids.to(model.device)).last_hidden_state.detach()
    return text_encoding


def sample_its_from_i0(model, i0, num_inference_steps=50):
    """
    Samples from P(i_1:T|i_0)
    """
    alpha_bar = model.scheduler.alphas_cumprod
    sqrt_one_minus_alpha_bar = (1 - alpha_bar) ** 0.5
    alphas = model.scheduler.alphas
    betas = 1 - alphas

    timesteps = model.scheduler.timesteps.to(model.device)
    t_to_idx = {int(v): k for k, v in enumerate(timesteps)}
    # Handle SDXL latent dimensions (128x128 instead of 64x64)
    latent_height = i0.shape[2]
    latent_width = i0.shape[3]
    its = torch.zeros(
        (num_inference_steps + 1, model.unet.config.in_channels, latent_height, latent_width)).to(i0.device)
    epsts = torch.zeros(
        (num_inference_steps, model.unet.config.in_channels, latent_height, latent_width)).to(i0.device)
    its[0] = i0
    for t in reversed(timesteps):
        idx = num_inference_steps - t_to_idx[int(t)]
        eps = torch.randn_like(i0).to(i0.device)
        its[idx] = i0 * (alpha_bar[t] ** 0.5) + eps * sqrt_one_minus_alpha_bar[t]
        epsts[idx - 1] = eps
    return its, epsts


def get_variance(model, timestep): #, prev_timestep):
    prev_timestep = timestep - model.scheduler.config.num_train_timesteps // model.scheduler.num_inference_steps
    alpha_prod_t = model.scheduler.alphas_cumprod[timestep]
    alpha_prod_t_prev = model.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else model.scheduler.final_alpha_cumprod
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev
    variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
    return variance


def reverse_step(model, model_output, timestep, sample, eta = 1, variance_noise=None):
    # 1. get previous step value (=t-1)
    prev_timestep = timestep - model.scheduler.config.num_train_timesteps // model.scheduler.num_inference_steps
    # 2. compute alphas, betas
    alpha_prod_t = model.scheduler.alphas_cumprod[timestep]
    alpha_prod_t_prev = model.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else model.scheduler.final_alpha_cumprod
    beta_prod_t = 1 - alpha_prod_t
    # 3. compute predicted original sample from predicted noise also called
    # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
    # 5. compute variance: "sigma_t(η)" -> see formula (16)
    # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
    # variance = self.scheduler._get_variance(timestep, prev_timestep)
    variance = get_variance(model, timestep) #, prev_timestep)
    std_dev_t = eta * variance ** (0.5)
    # Take care of asymetric reverse process (asyrp)
    model_output_direction = model_output
    # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    # pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * model_output_direction
    pred_sample_direction = (1 - alpha_prod_t_prev - eta * variance) ** (0.5) * model_output_direction
    # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction
    # 8. Add noice if eta > 0
    if eta > 0:
        if variance_noise is None:
            variance_noise = torch.randn(model_output.shape, device=model.device)
        sigma_z = eta * variance ** (0.5) * variance_noise
        prev_sample = prev_sample + sigma_z

    return prev_sample

def reverse_step_from_it(
    model, it, timestep, 
    cond_out=None, cond_embedding=None, uncond_out=None, uncond_embedding=None,
    cfg_scale=3.5, variance_noise=None,
    ):

    if uncond_out is None:
        uncond_out = model.unet.forward(it, timestep=timestep,
                                        encoder_hidden_states=uncond_embedding)

    if cond_out is None:
        cond_out = model.unet.forward(it, timestep=timestep,
                                        encoder_hidden_states=cond_embedding)

    noise_pred = uncond_out.sample + cfg_scale * (cond_out.sample - uncond_out.sample)

    it_prev = reverse_step(model, noise_pred, timestep, it, variance_noise=variance_noise)

    return it_prev

def initialize_theta_t(model, idx, method='base', expand_to_shape=None, theta_threshold=0.6, theta_init_low=0.25):
    if method == 'ones':
        init_val = 1.0
    elif method == 'mid':
        init_val = 0.5
    else:
        init_val = theta_init_low if idx > model.scheduler.num_inference_steps * theta_threshold else 1.0

    if expand_to_shape is not None:
        theta_t = torch.full(expand_to_shape, init_val, device=model.device, requires_grad=True)
    else:
        theta_t = torch.full((1,), init_val, device=model.device, requires_grad=True)
    return theta_t

def stack_embeddings(target_embedding, source_embedding, uncond_embedding):
    source_embedding_stack = torch.stack([uncond_embedding, source_embedding], dim=1)
    target_embedding_stack = torch.stack([uncond_embedding, target_embedding], dim=1)
    return source_embedding_stack, target_embedding_stack

