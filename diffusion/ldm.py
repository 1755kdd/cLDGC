import torch
import torch.nn as nn
import torch.optim as optim
import logging
import copy
import numpy as np
import networkx as nx
from torch.nn import functional as F
from tqdm import tqdm


class LatentDiffusion(DDPM):
    def __init__(self, first_stage_config, cond_stage_config, num_timesteps_cond=None, cond_stage_key="unconditional", first_stage_trainable=False,
                 cond_stage_trainable=False, concat_mode=False, cond_stage_forward=None, conditioning_key=None, scale_factor=1.0,
                 node_factor=1.0, edge_factor=1.0, recon_factor=0.0, graph_factor=1.0, scale_by_std=False, train_mode='sample', *args, **kwargs):
        self.num_timesteps_cond = default(num_timesteps_cond, 1)
        self.scale_by_std = scale_by_std
        assert cond_stage_key in cond_stage_key_list
        if conditioning_key is None:
            conditioning_key = 'concat' if concat_mode else 'crossattn'
        assert conditioning_key in conditioning_key_list
        self.cond_stage_key = cond_stage_key
        if self.cond_stage_key in ['masked_graph', 'prompt_graph', 'pe']:
            cond_stage_config = "__is_first_stage__"
        if cond_stage_config == '__is_unconditional__':
            conditioning_key = None
        ckpt_path = kwargs.pop("ckpt_path", None)
        ignore_keys = kwargs.pop("ignore_keys", [])
        super().__init__(conditioning_key=conditioning_key, *args, **kwargs)
        self.concat_mode = concat_mode
        self.first_stage_trainable = first_stage_trainable
        self.cond_stage_trainable = cond_stage_trainable
        try:
            self.num_downs = len(first_stage_config.params.ddconfig.ch_mult) - 1
        except:
            self.num_downs = 0
        if not scale_by_std:
            self.scale_factor = scale_factor
        else:
            self.register_buffer('scale_factor', torch.tensor(scale_factor))
        self.instantiate_first_stage(first_stage_config)
        self.instantiate_cond_stage(cond_stage_config)
        self.cond_stage_forward = cond_stage_forward
        self.clip_denoised = False
        self.bbox_tokenizer = None
        self.node_factor = cfg.diffusion.get("node_factor", node_factor)
        self.edge_factor = cfg.diffusion.get("edge_factor", edge_factor)
        self.graph_factor = cfg.diffusion.get("graph_factor", graph_factor)
        self.recon_factor = cfg.diffusion.get("recon_factor", recon_factor)
        assert train_mode in ['sample', 'half-sample', 'complete-noise']
        self.train_mode = train_mode
        self.restarted_from_ckpt = False
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys)
            self.restarted_from_ckpt = True

    def make_cond_schedule(self):
        self.cond_ids = torch.full(size=(self.num_timesteps,), fill_value=self.num_timesteps - 1, dtype=torch.long)
        ids = torch.round(torch.linspace(0, self.num_timesteps - 1, self.num_timesteps_cond)).long()
        self.cond_ids[:self.num_timesteps_cond] = ids

    @torch.no_grad()
    def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
        if self.scale_by_std and self.current_epoch == 0 and self.global_step == 0 and batch_idx == 0 and not self.restarted_from_ckpt:
            assert self.scale_factor == 1., 'rather not use custom rescaling and std-rescaling simultaneously'
            print("### USING STD-RESCALING ###")
            batch = super().get_input(batch)
            batch = batch.to(self.device)
            encoder_posterior = self.encode_first_stage(batch)
            z = self.get_first_stage_encoding(encoder_posterior).detach()
            del self.scale_factor
            self.register_buffer('scale_factor', 1. / z.flatten().std())
            print(f"setting self.scale_factor to {self.scale_factor}")
            print("### USING STD-RESCALING ###")

    def register_schedule(self, given_betas=None, beta_schedule="linear", timesteps=1000, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        super().register_schedule(given_betas, beta_schedule, timesteps, linear_start, linear_end, cosine_s)
        self.shorten_cond_schedule = self.num_timesteps_cond > 1
        if self.shorten_cond_schedule:
            self.make_cond_schedule()

    def instantiate_first_stage(self, config):
        model = eval(cfg.encoder.get("model_type", "GraphTransformerEncoder"))(cfg=cfg.encoder)
        self.first_stage_model = model
        sd = torch.load(config, map_location="cpu")
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        if "model_state" in list(sd.keys()):
            sd = sd["model_state"]
        state_dict = {}
        for k, v in sd.items():
            new_k = k[6:] if k.startswith('model.') else k
            state_dict[new_k] = v
        missing, unexpected = self.first_stage_model.load_state_dict(state_dict, strict=False)
        logging.info(f"Restored from {config} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            logging.info(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            logging.info(f"Unexpected Keys: {unexpected}")
        if not self.first_stage_trainable:
            self.first_stage_model = model.eval()
            self.first_stage_model.train = disabled_train
            for param in self.first_stage_model.parameters():
                param.requires_grad = False

    def instantiate_cond_stage(self, config):
        if config == "__is_first_stage__":
            print("Using first stage also as cond stage.")
            self.cond_stage_model = self.first_stage_model
        elif config == "__is_unconditional__":
            print(f"Training {self.__class__.__name__} as an unconditional model.")
            self.cond_stage_model = None
        else:
            model = eval(cfg.cond.get("model_type", "GraphTransformerEncoder"))(cfg=cfg.cond)
            self.cond_stage_model = model
            sd = torch.load(config, map_location="cpu")
            if "state_dict" in list(sd.keys()):
                sd = sd["state_dict"]
            if "model_state" in list(sd.keys()):
                sd = sd["model_state"]
            state_dict = {}
            for k, v in sd.items():
                new_k = k.replace('model.', '') if 'model' in k else k
                state_dict[new_k] = v
            missing, unexpected = self.first_stage_model.load_state_dict(state_dict, strict=False)
            print(f"Restored from {config} with {len(missing)} missing and {len(unexpected)} unexpected keys")
            if len(missing) > 0:
                print(f"Missing Keys: {missing}")
            if len(unexpected) > 0:
                print(f"Unexpected Keys: {unexpected}")
        if not self.cond_stage_trainable and self.cond_stage_model is not None:
            self.cond_stage_model.eval()
            self.cond_stage_model.train = disabled_train
            for param in self.cond_stage_model.parameters():
                param.requires_grad = False

    def get_first_stage_encoding(self, encoder_posterior):
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample()
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented")
        return self.scale_factor * z

    def get_learned_conditioning(self, c):
        if self.cond_stage_forward is None:
            if hasattr(self.cond_stage_model, 'encode') and callable(self.cond_stage_model.encode):
                c = self.cond_stage_model.encode(c)
                if isinstance(c, DiagonalGaussianDistribution):
                    c = c.mode()
            else:
                c = self.cond_stage_model(c)
        else:
            assert hasattr(self.cond_stage_model, self.cond_stage_forward)
            c = getattr(self.cond_stage_model, self.cond_stage_forward)(c)
        return c

    def get_weighting(self, h, w, Ly, Lx, device):
        weighting = self.delta_border(h, w)
        weighting = torch.clip(weighting, self.split_input_params["clip_min_weight"], self.split_input_params["clip_max_weight"])
        weighting = weighting.view(1, h * w, 1).repeat(1, 1, Ly * Lx).to(device)

        if self.split_input_params["tie_braker"]:
            L_weighting = self.delta_border(Ly, Lx)
            L_weighting = torch.clip(L_weighting, self.split_input_params["clip_min_tie_weight"], self.split_input_params["clip_max_tie_weight"])
            L_weighting = L_weighting.view(1, 1, Ly * Lx).to(device)
            weighting = weighting * L_weighting
        return weighting

    def get_fold_unfold(self, x, kernel_size, stride, uf=1, df=1):
        bs, nc, h, w = x.shape
        Ly = (h - kernel_size[0]) // stride[0] + 1
        Lx = (w - kernel_size[1]) // stride[1] + 1

        if uf == 1 and df == 1:
            fold_params = dict(kernel_size=kernel_size, dilation=1, padding=0, stride=stride)
            unfold = torch.nn.Unfold(**fold_params)
            fold = torch.nn.Fold(output_size=x.shape[2:], **fold_params)
            weighting = self.get_weighting(kernel_size[0], kernel_size[1], Ly, Lx, x.device).to(x.dtype)
            normalization = fold(weighting).view(1, 1, h, w)
            weighting = weighting.view((1, 1, kernel_size[0], kernel_size[1], Ly * Lx))
        elif uf > 1 and df == 1:
            fold_params = dict(kernel_size=kernel_size, dilation=1, padding=0, stride=stride)
            unfold = torch.nn.Unfold(**fold_params)

            fold_params2 = dict(kernel_size=(kernel_size[0] * uf, kernel_size[0] * uf),
                                dilation=1, padding=0,
                                stride=(stride[0] * uf, stride[1] * uf))
            fold = torch.nn.Fold(output_size=(x.shape[2] * uf, x.shape[3] * uf), **fold_params2)

            weighting = self.get_weighting(kernel_size[0] * uf, kernel_size[1] * uf, Ly, Lx, x.device).to(x.dtype)
            normalization = fold(weighting).view(1, 1, h * uf, w * uf)
            weighting = weighting.view((1, 1, kernel_size[0] * uf, kernel_size[1] * uf, Ly * Lx))

        elif df > 1 and uf == 1:
            fold_params = dict(kernel_size=kernel_size, dilation=1, padding=0, stride=stride)
            unfold = torch.nn.Unfold(**fold_params)

            fold_params2 = dict(kernel_size=(kernel_size[0] // df, kernel_size[0] // df),
                                dilation=1, padding=0,
                                stride=(stride[0] // df, stride[1] // df))
            fold = torch.nn.Fold(output_size=(x.shape[2] // df, x.shape[3] // df), **fold_params2)

            weighting = self.get_weighting(kernel_size[0] // df, kernel_size[1] // df, Ly, Lx, x.device).to(x.dtype)
            normalization = fold(weighting).view(1, 1, h // df, w // df)
            weighting = weighting.view((1, 1, kernel_size[0] // df, kernel_size[1] // df, Ly * Lx))

        else:
            raise NotImplementedError

        return fold, unfold, normalization, weighting

    def forward(self, batch, t=None, *args, **kwargs):
        if t is None:
            t = torch.randint(0, self.num_timesteps, (batch.num_graphs,), device=self.device).long()
        c = batch.get("c", None)
        return self.p_losses(batch, c, t, batch.batch_idx, *args, **kwargs)
    
    def compute_loss(self, graph_decode, batch_y):
        graph_criterion = nn.MSELoss()
        loss_task = graph_criterion(graph_decode, batch_y)
        return loss_task, graph_decode


class ConditionalDiffusion(nn.Module):
    def __init__(self, feat_dim, label_dim, hidden_dim=256, timesteps=1000):
        super().__init__()
        self.timesteps = timesteps
        
        self.register_buffer('betas', torch.linspace(1e-4, 0.02, self.timesteps))
        self.register_buffer('alphas', 1. - self.betas)
        self.register_buffer('alpha_bars', torch.cumprod(self.alphas, dim=0))
        
        self.node_net = nn.Sequential(
            nn.Linear(feat_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feat_dim)
        )
        self.edge_net = nn.Sequential(
            nn.Linear(feat_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feat_dim)
        )

    def forward(self, x, h, y, t):
        pass

    def diffuse(self, x, h, y):
        device = x.device
        batch_size = x.size(0)
        
        if h.size(0) != batch_size:
            h = h[torch.arange(batch_size) % h.size(0)] 
        
        t = torch.randint(0, self.timesteps, (batch_size,), device=device).long()
        alpha_bars = self.alpha_bars[t][:, None]
        
        sqrt_alpha = torch.sqrt(alpha_bars)
        sqrt_one_minus = torch.sqrt(1. - alpha_bars)
        
        eps_x = torch.randn_like(x)
        eps_h = torch.randn_like(h)
        
        x_noisy = sqrt_alpha * x + sqrt_one_minus * eps_x
        h_noisy = sqrt_alpha * h + sqrt_one_minus * eps_h
        
        return x_noisy, h_noisy, eps_x, eps_h, t
