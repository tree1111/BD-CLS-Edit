import torch
import torch.nn as nn
from torchvision.utils import save_image, make_grid
from torchvision import transforms
from torch.utils.data import DataLoader
from src.ds.cmnistbar_data_loader import ColoredBarMNIST
from tqdm import tqdm
import os
from src.colormnistbar.fi.models import ContextUnet


class FiPipline(nn.Module):
    def __init__(self,
                 c_size=16,
                 device=None,
                 save_path='out/color-bar-mnist/ddpm/exp_name', 
                 dat_path='dat/img/',
                 checkpoint_name=None):
        super(FiPipline, self).__init__()
        self.device = device
        self.checkpoint_name = checkpoint_name
        self.save_path = save_path
        self.dat_path = dat_path
        self.c_size = c_size

        nn_model = ContextUnet(in_channels=3, height=28,
                               width=28, n_feat=64, n_cfeat=c_size, n_downs=2).to(device=device)
        if checkpoint_name:
            checkpoint = torch.load(os.path.join(save_path, "saved_models", checkpoint_name), map_location=device)
            nn_model.to(device)
            nn_model.load_state_dict(checkpoint["model_state_dict"])
        self.nn_model = nn_model

    def train(self, batch_size=64, n_epoch=32, lr=1e-3, timesteps=500, beta1=1e-4, beta2=0.02):
        """Trains model for given inputs"""
        self.nn_model.train()
        _, _, ab_t = self.get_ddpm_noise_schedule(timesteps, beta1, beta2, self.device)
        dataset = self.instantiate_dataset(train=True)
        dataloader = self.initialize_dataloader(dataset, batch_size, self.checkpoint_name, self.save_path)
        optim = self.initialize_optimizer(self.nn_model, lr, self.checkpoint_name, self.save_path, self.device)
        scheduler = self.initialize_scheduler(optim, self.checkpoint_name, self.save_path, self.device)

        for epoch in range(self.get_start_epoch(self.checkpoint_name, self.save_path),
                           self.get_start_epoch(self.checkpoint_name, self.save_path) + n_epoch):
            ave_loss = 0

            for x, c in tqdm(dataloader, mininterval=2, desc=f"Epoch {epoch}"):
                x = x.to(self.device)
                c = c[:, :self.c_size]
                c = self.get_masked_context(c).to(self.device)

                # perturb data
                noise = torch.randn_like(x)
                t = torch.randint(1, timesteps + 1, (x.shape[0],)).to(self.device)
                x_pert = self.perturb_input(x, t, noise, ab_t)

                # predict noise
                pred_noise = self.nn_model(x_pert, t / timesteps, c=c)

                # obtain loss
                loss = torch.nn.functional.mse_loss(pred_noise, noise)

                # update params
                optim.zero_grad()
                loss.backward()
                optim.step()

                ave_loss += loss.item() / len(dataloader)
            scheduler.step()
            print(f"Epoch: {epoch}, loss: {ave_loss}")
            image_save_dir = self.save_path + '/saved_images'

            if not os.path.exists(image_save_dir):
                os.makedirs(image_save_dir)
            self.save_tensor_images(x, x_pert, self.get_x_unpert(x_pert, t, pred_noise, ab_t),
                                    epoch, image_save_dir)

            if (epoch + 1) % 10 == 0:
                x, _, _ = self.sample_ddpm(batch_size, context=c,
                                  timesteps=timesteps,
                                  beta1=beta1, beta2=beta2,
                                  save_rate=20,
                                  inference_transform=lambda x: (x + 1) / 2)
                save_image(
                    x[:64, :, :, :],
                    image_save_dir + f"/epoch_{epoch}_samples.png",
                    # value_range=(-1, 1)
                )
                self.nn_model.train()

            if (epoch+1) % 5 == 0 and epoch > 80:
                checkpoint_save_dir = self.save_path + '/saved_models'
                if not os.path.exists(checkpoint_save_dir):
                    os.makedirs(checkpoint_save_dir)
                self.save_checkpoint(self.nn_model, optim, scheduler, epoch, ave_loss,
                                     timesteps, beta1, beta2, self.device,
                                     dataloader.batch_size, checkpoint_save_dir, n_epoch)

    @torch.no_grad()
    def sample_ddpm(self, n_samples, iT=None, zs=None, context=None, timesteps=None,
                    beta1=None, beta2=None, save_rate=20, inference_transform=lambda x: (x + 1) / 2):
        """Returns the final denoised sample x0,
        intermediate samples xT, xT-1, ..., x1, and
        times tT, tT-1, ..., t1
        """
        if all([timesteps, beta1, beta2]):
            a_t, b_t, ab_t = self.get_ddpm_noise_schedule(timesteps, beta1, beta2, self.device)
        else:
            timesteps, a_t, b_t, ab_t = self.get_ddpm_params_from_checkpoint(self.save_path,
                                                                             self.checkpoint_name,
                                                                             self.device)

        self.nn_model.eval()
        if iT is not None:
            samples = iT
        else:
            samples = torch.randn((n_samples, self.nn_model.in_channels,
                                  self.nn_model.height, self.nn_model.width),
                                  device=self.device)
        intermediate_samples = [samples.detach().cpu()]  # samples at T = timesteps
        t_steps = [timesteps]  # keep record of time to use in animation generation
        for t in range(timesteps, 0, -1):
            print(f"Sampling timestep {t}", end="\r")
            if t % 50 == 0: print(f"Sampling timestep {t}")

            if zs is not None:
                z = zs[:, t-1, :, :, :] if t > 1 else 0
            else:
                z = torch.randn_like(samples) if t > 1 else 0

            pred_noise = self.nn_model(samples,
                                       torch.tensor([t / timesteps], device=self.device)[:, None, None, None],
                                       context)
            samples = self.denoise_add_noise(samples, t, pred_noise, a_t, b_t, ab_t, z)

            if t % save_rate == 1 or t < 8:
                intermediate_samples.append(inference_transform(samples.detach().cpu()))
                t_steps.append(t - 1)
        return intermediate_samples[-1], intermediate_samples, t_steps

    def perturb_input(self, x, t, noise, ab_t):
        """Perturbs given input
        i.e., Algorithm 1, step 5, argument of epsilon_theta in the article
        """
        return ab_t.sqrt()[t, None, None, None] * x + (1 - ab_t[t, None, None, None]).sqrt() * noise

    def instantiate_dataset(self, train=True):
        transform = transforms.Compose([
            transforms.ToTensor(),
            lambda x: 2 * (x - 0.5)
        ])
        if train:
            dat_set = ColoredBarMNIST(cg='full-ncm', root=self.dat_path, env='train', transform=transform)
        else:
            dat_set = ColoredBarMNIST(cg='full-ncm', root=self.dat_path, env='test', transform=transform)
        return dat_set


    def get_x_unpert(self, x_pert, t, pred_noise, ab_t):
        """Removes predicted noise pred_noise from perturbed image x_pert"""
        return (x_pert - (1 - ab_t[t, None, None, None]).sqrt() * pred_noise) / ab_t.sqrt()[t, None, None, None]


    def save_checkpoint(self, model, optimizer, scheduler, epoch, loss,
                        timesteps, beta1, beta2, device, batch_size,
                        save_dir, n_epoch):
        """Saves checkpoint for given variables"""
        if epoch < n_epoch - 1:
            fpath = os.path.join(save_dir, f"checkpoint_{epoch}.pth")
        else:
            fpath = os.path.join(save_dir, f"best.pth")

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "loss": loss,
            "timesteps": timesteps,
            "beta1": beta1,
            "beta2": beta2,
            "device": device,
            "batch_size": batch_size
        }
        torch.save(checkpoint, fpath)

    def initialize_optimizer(self, nn_model, lr, checkpoint_name, save_path, device):
        """Instantiates and initializes the optimizer based on checkpoint availability"""
        optim = torch.optim.Adam(nn_model.parameters(), lr=lr)
        if checkpoint_name:
            checkpoint = torch.load(os.path.join(save_path, "saved_models", checkpoint_name), map_location=device)
            optim.load_state_dict(checkpoint["optimizer_state_dict"])
        return optim

    def initialize_scheduler(self, optimizer, checkpoint_name, save_path, device):
        """Instantiates and initializes scheduler based on checkpoint availability"""
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1,
                                                      end_factor=0.01, total_iters=50)
        if checkpoint_name:
            checkpoint = torch.load(os.path.join(save_path, "saved_models", checkpoint_name), map_location=device)
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        return scheduler

    def get_start_epoch(self, checkpoint_name, save_path):
        """Returns starting epoch for training"""
        if checkpoint_name:
            start_epoch = torch.load(os.path.join(save_path, "saved_models", checkpoint_name),
                                     map_location=torch.device("cpu"))["epoch"] + 1
        else:
            start_epoch = 0
        return start_epoch

    def save_tensor_images(self, x_orig, x_noised, x_denoised, cur_epoch, save_dir):
        """Saves given tensors as a single image"""

        fpath = os.path.join(save_dir, f"x_orig_noised_denoised_{cur_epoch}.jpeg")
        inference_transform = lambda x: (x + 1) / 2
        save_image([make_grid(inference_transform(img.detach())) for img in [x_orig, x_noised, x_denoised]], fpath)

    def get_ddpm_noise_schedule(self, timesteps, beta1, beta2, device):
        """Returns ddpm noise schedule variables, a_t, b_t, ab_t
        b_t: \beta_t
        a_t: \alpha_t
        ab_t \bar{\alpha}_t
        """
        b_t = torch.linspace(beta1, beta2, timesteps + 1, device=device)
        a_t = 1 - b_t
        ab_t = torch.cumprod(a_t, dim=0)
        return a_t, b_t, ab_t

    def get_ddpm_params_from_checkpoint(self, save_path, checkpoint_name, device):
        """Returns scheduler variables T, a_t, ab_t, and b_t from checkpoint"""
        checkpoint = torch.load(os.path.join(save_path, "saved_models", checkpoint_name), torch.device("cpu"))
        T = checkpoint["timesteps"]
        a_t, b_t, ab_t = self.get_ddpm_noise_schedule(T, checkpoint["beta1"], checkpoint["beta2"], device)
        return T, a_t, b_t, ab_t

    def denoise_add_noise(self, x, t, pred_noise, a_t, b_t, ab_t, z):
        """Removes predicted noise from x and adds gaussian noise z
        i.e., Algorithm 2, step 4 at the ddpm article
        """
        noise = b_t.sqrt()[t] * z
        denoised_x = (x - pred_noise * ((1 - a_t[t]) / (1 - ab_t[t]).sqrt())) / a_t[t].sqrt()
        return denoised_x + noise

    def initialize_dataloader(self, dataset, batch_size, checkpoint_name, save_path):
        """Returns dataloader based on batch-size of checkpoint if present"""
        if checkpoint_name:
            batch_size = torch.load(os.path.join(save_path, "saved_models", checkpoint_name),
                                    map_location=torch.device("cpu"))["batch_size"]
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                           drop_last=True, num_workers=8)

    def get_masked_context(self, context, p=0.9):
        "Randomly mask out context"
        return context * torch.bernoulli(torch.ones((context.shape[0], 1)) * p)



