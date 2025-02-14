import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from tqdm import tqdm
from functools import partial
from contextlib import contextmanager
from torchvision.utils import make_grid
from einops import rearrange

class DDPM(pl.LightningModule):
    def __init__(self, gt_config=None, timesteps=1000, beta_schedule="linear", loss_type="l2", 
                 ckpt_path=None, ignore_keys=[], load_only_unet=False, monitor="val/loss", 
                 use_ema=False, first_stage_key=None, image_size=256, hid_dim=3, log_every_t=100,
                 clip_denoised=False, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3, 
                 given_betas=None, original_elbo_weight=0., v_posterior=0., l_simple_weight=1.,
                 conditioning_key=None, parameterization="x0", scheduler_config=None,
                 use_positional_encodings=False, learn_logvar=False, logvar_init=0.):
        super().__init__()
        self.parameterization = parameterization
        self.clip_denoised = clip_denoised
        self.log_every_t = log_every_t
        self.image_size = image_size
        self.hid_dim = hid_dim
        self.force_undirected = gt_config.diffusion.get('force_undirected', False)
        self.use_positional_encodings = use_positional_encodings
        self.model = DiffusionWrapper(gt_config, conditioning_key)
        self.use_ema = use_ema
        
        if self.use_ema:
            self.model_ema = LitEma(self.model)

        self.use_scheduler = scheduler_config is not None
        if self.use_scheduler:
            self.scheduler_config = scheduler_config

        self.v_posterior = v_posterior
        self.original_elbo_weight = original_elbo_weight
        self.l_simple_weight = l_simple_weight

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys, only_model=load_only_unet)

        self.register_schedule(given_betas=given_betas, beta_schedule=beta_schedule, 
                             timesteps=timesteps, linear_start=linear_start, linear_end=linear_end,
                             cosine_s=cosine_s)

        self.loss_type = loss_type
        self.learn_logvar = learn_logvar
        self.logvar = nn.Parameter(torch.full((self.num_timesteps,), logvar_init), 
                                 requires_grad=learn_logvar)

    def register_schedule(self, given_betas=None, beta_schedule="linear", timesteps=1000,
                         linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        if given_betas is not None:
            betas = given_betas
        else:
            betas = make_beta_schedule(beta_schedule, timesteps, linear_start, linear_end, cosine_s)

        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end

        to_torch = partial(torch.tensor, dtype=torch.float32)
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
                    1. - alphas_cumprod) + self.v_posterior * betas
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        self.register_buffer('posterior_log_variance_clipped', 
                           to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

        if self.parameterization == "eps":
            lvlb_weights = self.betas ** 2 / (2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod))
        else:
            lvlb_weights = 0.5 * torch.sqrt(torch.Tensor(alphas_cumprod)) / (2. * 1 - torch.Tensor(alphas_cumprod))
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer('lvlb_weights', lvlb_weights, persistent=False)

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context:
                    print(f"{context}: Restored training weights")

    def init_from_ckpt(self, path, ignore_keys=list(), only_model=False):
        sd = torch.load(path, map_location="cpu").get("state_dict", )
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    del sd[k]
        (self.model if only_model else self).load_state_dict(sd, strict=False)

    def q_mean_variance(self, x_start, t, batch_idx):
        mean = extract_into_sparse_tensor(self.sqrt_alphas_cumprod, t, batch_idx) * x_start
        variance = extract_into_sparse_tensor(1.0 - self.alphas_cumprod, t, batch_idx)
        log_variance = extract_into_sparse_tensor(self.log_one_minus_alphas_cumprod, t, batch_idx)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise, batch_idx):
        return (extract_into_sparse_tensor(self.sqrt_recip_alphas_cumprod, t, batch_idx) * x_t 
              - extract_into_sparse_tensor(self.sqrt_recipm1_alphas_cumprod, t, batch_idx) * noise)

    def q_posterior(self, x_start, x_t, t, batch_idx):
        posterior_mean = (extract_into_sparse_tensor(self.posterior_mean_coef1, t, batch_idx) * x_start 
                        + extract_into_sparse_tensor(self.posterior_mean_coef2, t, batch_idx) * x_t)
        posterior_variance = extract_into_sparse_tensor(self.posterior_variance, t, batch_idx)
        posterior_log_variance = extract_into_sparse_tensor(self.posterior_log_variance_clipped, t, batch_idx)
        return posterior_mean, posterior_variance, posterior_log_variance

    def p_mean_variance(self, x, t, batch_idx, clip_denoised):
        model_out = self.model(x, t)
        x_recon = model_out if self.parameterization == "x0" else self.predict_start_from_noise(x, t, model_out, batch_idx)
        if clip_denoised:
            x_recon.clamp_(-1., 1.)
        return self.q_posterior(x_recon, x, t, batch_idx)

    @torch.no_grad()
    def p_sample(self, x, t, batch_idx, clip_denoised=True, repeat_noise=False):
        model_mean, _, model_log_variance = self.p_mean_variance(x, t, batch_idx, clip_denoised)
        noise = noise_like(x.shape, x.device, repeat_noise)
        nonzero_mask = (1 - (t == 0).float()).reshape(x.size(0), *((1,)*(x.ndim-1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, batch_idx, return_intermediates=False):
        device = self.betas.device
        img = torch.randn(shape, device=device)
        intermediates = [img]
        for i in tqdm(reversed(range(self.num_timesteps)), desc='Sampling t', total=self.num_timesteps):
            img = self.p_sample(img, torch.full((len(batch_idx.unique()),), i, device=device), batch_idx)
            if i % self.log_every_t == 0 or i == self.num_timesteps-1:
                intermediates.append(img)
        return (img, intermediates) if return_intermediates else img

    @torch.no_grad()
    def sample(self, batch_idx, batch_size=16, return_intermediates=False):
        return self.p_sample_loop((batch_idx.size(0), self.hid_dim), batch_idx, return_intermediates)

    def q_sample(self, x_start, t, batch_idx=None, noise=None):
        noise = noise if noise is not None else torch.randn_like(x_start)
        if batch_idx is None:
            return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start 
                  + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)
        return (extract_into_sparse_tensor(self.sqrt_alphas_cumprod, t, batch_idx) * x_start 
              + extract_into_sparse_tensor(self.sqrt_one_minus_alphas_cumprod, t, batch_idx) * noise)

    def get_loss(self, pred, target, mean=True):
        if self.loss_type == 'l1':
            loss = (target - pred).abs().mean() if mean else (target - pred).abs()
        elif self.loss_type == 'l2':
            loss = F.mse_loss(pred, target) if mean else F.mse_loss(pred, target, reduction='none')
        return loss

    def p_losses(self, batch, t, noise=None, task_id=None):
        x_start = torch.cat([batch.x, batch.edge_attr], dim=0)
        noise = noise if noise is not None else torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, batch.batch_idx, noise)
        batch.x, batch.edge_attr = x_noisy[:batch.num_nodes], x_noisy[batch.num_nodes:]
        model_out = self.model(batch, t, task_id)
        target = x_start if self.parameterization == "x0" else noise
        loss = self.get_loss(model_out, target, mean=False).mean(1)
        loss_simple = loss.mean() * self.l_simple_weight
        loss_vlb = (self.lvlb_weights[t] * loss).mean()
        return loss_simple + self.original_elbo_weight * loss_vlb, {
            'loss': loss_simple + self.original_elbo_weight * loss_vlb,
            'loss_simple': loss_simple,
            'loss_vlb': loss_vlb
        }

    def forward(self, batch, task_id=None):
        t = torch.randint(0, self.num_timesteps, (batch.num_graphs,), device=self.device)
        return self.p_losses(batch, t, task_id=task_id)

    def get_input(self, batch):
        batch.batch_idx = torch.cat([batch.batch, num2batch(batch.num_node_per_graph**2)])
        batch.batch_size = batch.num_graphs
        return batch

    def shared_step(self, batch, task_id=None):
        batch = self.get_input(batch)
        if task_id is not None:
            batch.task_id = task_id
        return self(batch, task_id)

    def training_step(self, batch, task_id=None):
        loss, loss_dict = self.shared_step(batch, task_id)
        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("global_step", self.global_step, prog_bar=True, logger=True)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, task_id=None):
        _, loss_dict = self.shared_step(batch, task_id)
        with self.ema_scope():
            _, ema_loss_dict = self.shared_step(batch, task_id)
        self.log_dict({**{k+'_ema':v for k,v in ema_loss_dict.items()}, **loss_dict}, 
                     logger=True, on_epoch=True)

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.model)

    def configure_optimizers(self):
        params = list(self.model.parameters()) + [self.logvar] if self.learn_logvar else self.model.parameters()
        return torch.optim.AdamW(params, lr=self.learning_rate)
    
    