import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_sum, scatter_mean
from tqdm.auto import tqdm
import pickle

from models.common import compose_context, ShiftedSoftplus
from models.egnn import EGNN
# from mamba_ssm import Mamba2

from models.uni_transformer import UniTransformerO2TwoUpdateGeneral
from models.Diff_function import *


def get_refine_net(refine_net_type, config):
    if refine_net_type == 'uni_o2':
        refine_net = UniTransformerO2TwoUpdateGeneral(
            num_blocks=config.num_blocks,
            num_layers=config.num_layers,
            hidden_dim=config.hidden_dim,
            n_heads=config.n_heads,
            k=config.knn,
            edge_feat_dim=config.edge_feat_dim,
            num_r_gaussian=config.num_r_gaussian,
            num_node_types=config.num_node_types,
            act_fn=config.act_fn,
            norm=config.norm,
            cutoff_mode=config.cutoff_mode,
            ew_net_type=config.ew_net_type,
            num_x2h=config.num_x2h,
            num_h2x=config.num_h2x,
            r_max=config.r_max,
            x2h_out_fc=config.x2h_out_fc,
            sync_twoup=config.sync_twoup
        )
    elif refine_net_type == 'egnn':
        refine_net = EGNN(
            num_layers=config.num_layers,
            hidden_dim=config.hidden_dim,
            edge_feat_dim=config.edge_feat_dim,
            num_r_gaussian=1,
            k=config.knn,
            cutoff_mode=config.cutoff_mode
        )
    else:
        raise ValueError(refine_net_type)
    return refine_net


class ScorePosNet3D(nn.Module):

    def __init__(self, config, protein_atom_feature_dim, ligand_atom_feature_dim):
        super().__init__()
        self.config = config

        # variance schedule
        self.model_mean_type = config.model_mean_type  # ['noise', 'C0']
        self.loss_v_weight = config.loss_v_weight

        self.sample_time_method = config.sample_time_method  # ['importance', 'symmetric']
        if config.beta_schedule == 'cosine':
            alphas = cosine_beta_schedule(config.num_diffusion_timesteps, config.pos_beta_s) ** 2
            betas = 1. - alphas
        else:
            betas = get_beta_schedule(
                beta_schedule=config.beta_schedule,
                beta_start=config.beta_start,
                beta_end=config.beta_end,
                num_diffusion_timesteps=config.num_diffusion_timesteps,
            )
            alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        self.betas = to_torch_const(betas)
        self.num_timesteps = self.betas.size(0)
        self.alphas_cumprod = to_torch_const(alphas_cumprod)
        self.alphas_cumprod_prev = to_torch_const(alphas_cumprod_prev)

        self.sqrt_alphas_cumprod = to_torch_const(np.sqrt(alphas_cumprod))
        self.sqrt_one_minus_alphas_cumprod = to_torch_const(np.sqrt(1. - alphas_cumprod))
        self.sqrt_recip_alphas_cumprod = to_torch_const(np.sqrt(1. / alphas_cumprod))
        self.sqrt_recipm1_alphas_cumprod = to_torch_const(np.sqrt(1. / alphas_cumprod - 1))

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.posterior_mean_c0_coef = to_torch_const(betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.posterior_mean_ct_coef = to_torch_const(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))
        self.posterior_var = to_torch_const(posterior_variance)
        self.posterior_logvar = to_torch_const(np.log(np.append(self.posterior_var[1], self.posterior_var[1:])))

        # atom type diffusion schedule in log space
        if config.v_beta_schedule == 'cosine':
            alphas_v = cosine_beta_schedule(self.num_timesteps, config.v_beta_s)
        else:
            raise NotImplementedError
        log_alphas_v = np.log(alphas_v)
        log_alphas_cumprod_v = np.cumsum(log_alphas_v)
        self.log_alphas_v = to_torch_const(log_alphas_v)
        self.log_one_minus_alphas_v = to_torch_const(log_1_min_a(log_alphas_v))
        self.log_alphas_cumprod_v = to_torch_const(log_alphas_cumprod_v)
        self.log_one_minus_alphas_cumprod_v = to_torch_const(log_1_min_a(log_alphas_cumprod_v))

        self.register_buffer('Lt_history', torch.zeros(self.num_timesteps))
        self.register_buffer('Lt_count', torch.zeros(self.num_timesteps))

        # model definition
        self.hidden_dim = config.hidden_dim
        self.num_classes = ligand_atom_feature_dim
        emb_dim = self.hidden_dim - 1 if self.config.node_indicator else self.hidden_dim

        # atom embedding
        self.protein_atom_emb = nn.Linear(protein_atom_feature_dim, emb_dim)

        # center pos
        self.center_pos_mode = config.center_pos_mode  # ['none', 'protein']

        # time embedding
        self.time_emb_dim = config.time_emb_dim
        self.time_emb_mode = config.time_emb_mode  # ['simple', 'sin']
        if self.time_emb_dim > 0:
            if self.time_emb_mode == 'simple':
                self.ligand_atom_emb = nn.Linear(ligand_atom_feature_dim + 1, emb_dim)
            elif self.time_emb_mode == 'sin':
                self.time_emb = nn.Sequential(
                    SinusoidalPosEmb(self.time_emb_dim),
                    nn.Linear(self.time_emb_dim, self.time_emb_dim * 4),
                    nn.GELU(),
                    nn.Linear(self.time_emb_dim * 4, self.time_emb_dim)
                )
                self.ligand_atom_emb = nn.Linear(ligand_atom_feature_dim + self.time_emb_dim, emb_dim)
            else:
                raise NotImplementedError
        else:
            self.ligand_atom_emb = nn.Linear(ligand_atom_feature_dim, emb_dim)

        self.refine_net_type = config.model_type
        self.refine_net = get_refine_net(self.refine_net_type, config)

        self.v_inference = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            ShiftedSoftplus(),
            nn.Linear(self.hidden_dim, ligand_atom_feature_dim),
        )

        self.use_quality_head = True
        self.quality_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            ShiftedSoftplus(),
            nn.Linear(self.hidden_dim, 1)
        )

        # 
        self.info_dict = None
        info_path = getattr(config, "info_path", None)
        if info_path is not None:
            with open(info_path, "rb") as f:
                self.info_dict = pickle.load(f)
            print(f"[ScorePosNet3D] Loaded info.pkl from {info_path}, total keys = {len(self.info_dict)}")

        # ---- PRIOR CONFIG ----
        self.prior_dim = 3
        self.use_prior = self.info_dict is not None
        self.dropout = 0.1

        # prior
        self.use_prior_norm = getattr(config, "use_prior_norm", False)
        prior_mean = getattr(config, "prior_mean", None)  # e.g. [mean_qed, mean_sa, mean_vina]
        prior_std = getattr(config, "prior_std", None)    # e.g. [std_qed, std_sa, std_vina]
        if self.use_prior_norm:
            if prior_mean is None or prior_std is None:
                raise ValueError("use_prior_norm=True but config.prior_mean / config.prior_std not provided.")
            self.register_buffer("prior_mean", torch.tensor(prior_mean, dtype=torch.float32).view(1, -1))
            self.register_buffer("prior_std", torch.tensor(prior_std, dtype=torch.float32).view(1, -1))

        if self.use_prior or getattr(config, "allow_prior_vec_without_info", True):
            # 
            self.prior_proj = nn.Sequential(
                nn.Linear(self.prior_dim, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
            self.prior_gate = nn.Sequential(
                nn.LayerNorm(self.hidden_dim * 2),
                nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, 1),
            )
            self.prior_only_ligand = getattr(config, "prior_only_ligand", True)

    # ------------------ PRIOR UTILS ------------------
    def _normalize_prior(self, prior_vec: torch.Tensor) -> torch.Tensor:
        if not self.use_prior_norm:
            return prior_vec
        return (prior_vec - self.prior_mean.to(prior_vec.device)) / (self.prior_std.to(prior_vec.device) + 1e-8)

    def _prior_candidate_keys(self, pid: str, lid: str):

        keys = []

        pid = str(pid)
        lid = str(lid)


        keys.append(f"{pid}-{lid}")

        if not pid.startswith("test-"):
            keys.append(f"test-{pid}-{lid}")


        if pid.startswith("test-"):
            keys.append(f"{pid[len('test-'):]}-{lid}")

        if lid.startswith("test-"):
            keys.append(f"{pid}-{lid[len('test-'):]}")
            if not pid.startswith("test-"):
                keys.append(f"test-{pid}-{lid[len('test-'):]}")
        else:
            keys.append(f"{pid}-test-{lid}")
            if not pid.startswith("test-"):
                keys.append(f"test-{pid}-test-{lid}")

        
        seen = set()
        out = []
        for k in keys:
            if k not in seen:
                out.append(k)
                seen.add(k)
        return out

    def _build_global_prior(self, pid_list, lid_list, device):
        assert self.info_dict is not None, "info_dict is None but _build_global_prior was called."

        B = len(pid_list)
        prior = torch.zeros(B, self.prior_dim, device=device)

        
        default_prior = getattr(self, "default_prior", None)
        if default_prior is None:
            default_prior = torch.zeros(1, self.prior_dim, device=device)

        miss = 0
        for i in range(B):
            pid = pid_list[i]
            lid = lid_list[i]

            info = None
            for key in self._prior_candidate_keys(pid, lid):
                if key in self.info_dict:
                    info = self.info_dict[key]
                    break

            if info is None:
                prior[i] = default_prior.to(device)
                miss += 1
                continue

            prior[i, 0] = float(info["QED"])
            prior[i, 1] = float(info["SA"])
            prior[i, 2] = -float(info["affinity_vina"])  # vina = -affinity



        return prior

    def _apply_prior_to_nodes(self, h_all, batch_all, mask_ligand, prior_vec):
        """
        h_all: [N_total, D]
        prior_vec: [B, prior_dim]  
        """
        prior_vec = self._normalize_prior(prior_vec)
        prior_emb = self.prior_proj(prior_vec)      # [B, D]
        prior_node = prior_emb[batch_all]           # [N_total, D]

        if self.prior_only_ligand:
            prior_node = prior_node * mask_ligand.float().unsqueeze(-1)

        gate = torch.sigmoid(self.prior_gate(torch.cat([h_all, prior_node], dim=-1)))  # [N_total, 1]
        h_all = h_all + gate * prior_node
        return h_all

    # ------------------ FORWARD ------------------
    def forward(
        self,
        protein_pos, protein_v, batch_protein,
        init_ligand_pos, init_ligand_v, batch_ligand,
        time_step=None, return_all=False, fix_x=False,
        pid=None, lid=None, prior_vec=None
    ):
        """
        prior_vec: [B,3] -> [QED, SA, Vina]
        """
        num_graphs = int(batch_protein.max().item()) + 1

        init_ligand_v = F.one_hot(init_ligand_v, self.num_classes).float()

        # time embedding
        if self.time_emb_dim > 0:
            if self.time_emb_mode == 'simple':
                input_ligand_feat = torch.cat([
                    init_ligand_v,
                    (time_step / self.num_timesteps)[batch_ligand].unsqueeze(-1)
                ], -1)
            elif self.time_emb_mode == 'sin':
                time_feat = self.time_emb(time_step)
                input_ligand_feat = torch.cat([init_ligand_v, time_feat], -1)
            else:
                raise NotImplementedError
        else:
            input_ligand_feat = init_ligand_v

        h_protein = self.protein_atom_emb(protein_v)
        init_ligand_h = self.ligand_atom_emb(input_ligand_feat)

        if self.config.node_indicator:
            h_protein = torch.cat([h_protein, torch.zeros(len(h_protein), 1, device=h_protein.device)], -1)
            init_ligand_h = torch.cat([init_ligand_h, torch.ones(len(init_ligand_h), 1, device=h_protein.device)], -1)

        h_all, pos_all, batch_all, mask_ligand = compose_context(
            h_protein=h_protein,
            h_ligand=init_ligand_h,
            pos_protein=protein_pos,
            pos_ligand=init_ligand_pos,
            batch_protein=batch_protein,
            batch_ligand=batch_ligand,
        )

        # ---- PRIOR INJECTION ----
        if prior_vec is not None:
            assert prior_vec.shape == (num_graphs, self.prior_dim), f"prior_vec shape should be {(num_graphs, self.prior_dim)}"

            if torch.rand(1).item() < self.dropout:
                prior_vec = torch.zeros_like(prior_vec)

            prior_vec = prior_vec.to(h_all.device)
            h_all = self._apply_prior_to_nodes(h_all, batch_all, mask_ligand, prior_vec)

        elif (self.use_prior and (pid is not None) and (lid is not None)):
            assert len(pid) == num_graphs and len(lid) == num_graphs
            prior_vec2 = self._build_global_prior(pid, lid, device=h_all.device)  # [B,3]

            if torch.rand(1).item() < self.dropout:
                prior_vec2 = torch.zeros_like(prior_vec2)

            h_all = self._apply_prior_to_nodes(h_all, batch_all, mask_ligand, prior_vec2)

        outputs = self.refine_net(h_all, pos_all, mask_ligand, batch_all, return_all=return_all, fix_x=fix_x)
        final_pos, final_h = outputs['x'], outputs['h']
        final_ligand_pos, final_ligand_h = final_pos[mask_ligand], final_h[mask_ligand]
        final_ligand_v = self.v_inference(final_ligand_h)

        preds = {
            'pred_ligand_pos': final_ligand_pos,
            'pred_ligand_v': final_ligand_v,
            'final_h': final_h,
            'final_ligand_h': final_ligand_h
        }

        if return_all:
            final_all_pos, final_all_h = outputs['all_x'], outputs['all_h']
            final_all_ligand_pos = [pos[mask_ligand] for pos in final_all_pos]
            final_all_ligand_v = [self.v_inference(h[mask_ligand]) for h in final_all_h]
            preds.update({
                'layer_pred_ligand_pos': final_all_ligand_pos,
                'layer_pred_ligand_v': final_all_ligand_v
            })
        return preds

    # ------------------ diffusion utils ------------------
    def q_v_pred_one_timestep(self, log_vt_1, t, batch):
        log_alpha_t = extract(self.log_alphas_v, t, batch)
        log_1_min_alpha_t = extract(self.log_one_minus_alphas_v, t, batch)
        log_probs = log_add_exp(
            log_vt_1 + log_alpha_t,
            log_1_min_alpha_t - np.log(self.num_classes)
        )
        return log_probs

    def q_v_pred(self, log_v0, t, batch):
        log_cumprod_alpha_t = extract(self.log_alphas_cumprod_v, t, batch)
        log_1_min_cumprod_alpha = extract(self.log_one_minus_alphas_cumprod_v, t, batch)
        log_probs = log_add_exp(
            log_v0 + log_cumprod_alpha_t,
            log_1_min_cumprod_alpha - np.log(self.num_classes)
        )
        return log_probs

    def q_v_sample(self, log_v0, t, batch):
        log_qvt_v0 = self.q_v_pred(log_v0, t, batch)
        sample_index = log_sample_categorical(log_qvt_v0)
        log_sample = index_to_log_onehot(sample_index, self.num_classes)
        return sample_index, log_sample

    def q_v_posterior(self, log_v0, log_vt, t, batch):
        t_minus_1 = t - 1
        t_minus_1 = torch.where(t_minus_1 < 0, torch.zeros_like(t_minus_1), t_minus_1)
        log_qvt1_v0 = self.q_v_pred(log_v0, t_minus_1, batch)
        unnormed_logprobs = log_qvt1_v0 + self.q_v_pred_one_timestep(log_vt, t, batch)
        log_vt1_given_vt_v0 = unnormed_logprobs - torch.logsumexp(unnormed_logprobs, dim=-1, keepdim=True)
        return log_vt1_given_vt_v0

    def kl_v_prior(self, log_x_start, batch):
        num_graphs = batch.max().item() + 1
        log_qxT_prob = self.q_v_pred(log_x_start, t=[self.num_timesteps - 1] * num_graphs, batch=batch)
        log_half_prob = -torch.log(self.num_classes * torch.ones_like(log_qxT_prob))
        kl_prior = categorical_kl(log_qxT_prob, log_half_prob)
        kl_prior = scatter_mean(kl_prior, batch, dim=0)
        return kl_prior

    def _predict_x0_from_eps(self, xt, eps, t, batch):
        pos0_from_e = extract(self.sqrt_recip_alphas_cumprod, t, batch) * xt - \
                      extract(self.sqrt_recipm1_alphas_cumprod, t, batch) * eps
        return pos0_from_e

    def q_pos_posterior(self, x0, xt, t, batch):
        pos_model_mean = extract(self.posterior_mean_c0_coef, t, batch) * x0 + \
                         extract(self.posterior_mean_ct_coef, t, batch) * xt
        return pos_model_mean

    def kl_pos_prior(self, pos0, batch):
        num_graphs = batch.max().item() + 1
        a_pos = extract(self.alphas_cumprod, [self.num_timesteps - 1] * num_graphs, batch)
        pos_model_mean = a_pos.sqrt() * pos0
        pos_log_variance = torch.log((1.0 - a_pos).sqrt())
        kl_prior = normal_kl(torch.zeros_like(pos_model_mean), torch.zeros_like(pos_log_variance),
                             pos_model_mean, pos_log_variance)
        kl_prior = scatter_mean(kl_prior, batch, dim=0)
        return kl_prior

    def sample_time(self, num_graphs, device, method):
        if method == 'importance':
            if not (self.Lt_count > 10).all():
                return self.sample_time(num_graphs, device, method='symmetric')

            Lt_sqrt = torch.sqrt(self.Lt_history + 1e-10) + 0.0001
            Lt_sqrt[0] = Lt_sqrt[1]
            pt_all = Lt_sqrt / Lt_sqrt.sum()

            time_step = torch.multinomial(pt_all, num_samples=num_graphs, replacement=True)
            pt = pt_all.gather(dim=0, index=time_step)
            return time_step, pt

        elif method == 'symmetric':
            time_step = torch.randint(0, self.num_timesteps, size=(num_graphs // 2 + 1,), device=device)
            time_step = torch.cat([time_step, self.num_timesteps - time_step - 1], dim=0)[:num_graphs]
            pt = torch.ones_like(time_step).float() / self.num_timesteps
            return time_step, pt

        else:
            raise ValueError

    def compute_pos_Lt(self, pos_model_mean, x0, xt, t, batch):
        pos_log_variance = extract(self.posterior_logvar, t, batch)
        pos_true_mean = self.q_pos_posterior(x0=x0, xt=xt, t=t, batch=batch)
        kl_pos = normal_kl(pos_true_mean, pos_log_variance, pos_model_mean, pos_log_variance)
        kl_pos = kl_pos / np.log(2.)

        decoder_nll_pos = -log_normal(x0, means=pos_model_mean, log_scales=0.5 * pos_log_variance)
        mask = (t == 0).float()[batch]
        loss_pos = scatter_mean(mask * decoder_nll_pos + (1. - mask) * kl_pos, batch, dim=0)
        return loss_pos

    def compute_v_Lt(self, log_v_model_prob, log_v0, log_v_true_prob, t, batch):
        kl_v = categorical_kl(log_v_true_prob, log_v_model_prob)
        decoder_nll_v = -log_categorical(log_v0, log_v_model_prob)
        mask = (t == 0).float()[batch]
        loss_v = scatter_mean(mask * decoder_nll_v + (1. - mask) * kl_v, batch, dim=0)
        return loss_v

    # ------------------ TRAIN LOSS (加 prior_vec 支持) ------------------
    def get_diffusion_loss(
        self, protein_pos, protein_v, batch_protein,
        ligand_pos, ligand_v, batch_ligand,
        time_step=None, pid=None, lid=None, prior_vec=None
    ):
        num_graphs = batch_protein.max().item() + 1
        protein_pos, ligand_pos, _ = center_pos(
            protein_pos, ligand_pos, batch_protein, batch_ligand, mode=self.center_pos_mode)

        # 1. sample noise levels
        if time_step is None:
            time_step, pt = self.sample_time(num_graphs, protein_pos.device, self.sample_time_method)
        else:
            pt = torch.ones_like(time_step).float() / self.num_timesteps
        a = self.alphas_cumprod.index_select(0, time_step)

        # 2. perturb pos and v
        a_pos = a[batch_ligand].unsqueeze(-1)
        pos_noise = torch.randn_like(ligand_pos)
        ligand_pos_perturbed = a_pos.sqrt() * ligand_pos + (1.0 - a_pos).sqrt() * pos_noise

        log_ligand_v0 = index_to_log_onehot(ligand_v, self.num_classes)
        ligand_v_perturbed, log_ligand_vt = self.q_v_sample(log_ligand_v0, time_step, batch_ligand)

        # 3. forward
        preds = self(
            protein_pos=protein_pos,
            protein_v=protein_v,
            batch_protein=batch_protein,
            init_ligand_pos=ligand_pos_perturbed,
            init_ligand_v=ligand_v_perturbed,
            batch_ligand=batch_ligand,
            time_step=time_step,
            pid=pid,
            lid=lid,
            prior_vec=prior_vec,
        )

        pred_ligand_pos, pred_ligand_v = preds['pred_ligand_pos'], preds['pred_ligand_v']
        pred_pos_noise = pred_ligand_pos - ligand_pos_perturbed

        if self.model_mean_type == 'noise':
            pos0_from_e = self._predict_x0_from_eps(
                xt=ligand_pos_perturbed, eps=pred_pos_noise, t=time_step, batch=batch_ligand)
            pos_model_mean = self.q_pos_posterior(
                x0=pos0_from_e, xt=ligand_pos_perturbed, t=time_step, batch=batch_ligand)
        elif self.model_mean_type == 'C0':
            pos_model_mean = self.q_pos_posterior(
                x0=pred_ligand_pos, xt=ligand_pos_perturbed, t=time_step, batch=batch_ligand)
        else:
            raise ValueError

        # pos loss
        if self.model_mean_type == 'C0':
            target, pred = ligand_pos, pred_ligand_pos
        else:
            target, pred = pos_noise, pred_pos_noise
        loss_pos = scatter_mean(((pred - target) ** 2).sum(-1), batch_ligand, dim=0).mean()

        # type loss
        log_ligand_v_recon = F.log_softmax(pred_ligand_v, dim=-1)
        log_v_model_prob = self.q_v_posterior(log_ligand_v_recon, log_ligand_vt, time_step, batch_ligand)
        log_v_true_prob = self.q_v_posterior(log_ligand_v0, log_ligand_vt, time_step, batch_ligand)
        kl_v = self.compute_v_Lt(
            log_v_model_prob=log_v_model_prob, log_v0=log_ligand_v0,
            log_v_true_prob=log_v_true_prob, t=time_step, batch=batch_ligand
        )
        loss_v = kl_v.mean()
        loss = loss_pos + loss_v * self.loss_v_weight

        return {
            'loss_pos': loss_pos,
            'loss_v': loss_v,
            'loss': loss,
            'x0': ligand_pos,
            'pred_ligand_pos': pred_ligand_pos,
            'pred_ligand_v': pred_ligand_v,
            'pred_pos_noise': pred_pos_noise,
            'ligand_v_recon': F.softmax(pred_ligand_v, dim=-1)
        }

    # ------------------ SAMPLING ( prior_vec + prior CFG) ------------------
    @torch.no_grad()
    def sample_diffusion(
        self,
        protein_pos, protein_v, batch_protein,
        init_ligand_pos, init_ligand_v, batch_ligand,
        num_steps=None, center_pos_mode=None, pos_only=False,
        prior_vec=None,
        prior_guidance_scale=0.0,   # 
        prior_uncond_mode="zero",   # "zero" or "mean"
    ):
        """
        prior_vec: [B,3] 采样想要的属性
        prior_guidance_scale: w，常用 1~5 之间试
        """
        if num_steps is None:
            num_steps = self.num_timesteps

        num_graphs = batch_protein.max().item() + 1
        device = protein_pos.device

        # center
        protein_pos, init_ligand_pos, offset = center_pos(
            protein_pos, init_ligand_pos, batch_protein, batch_ligand, mode=center_pos_mode
        )

        # prepare prior
        if prior_vec is not None:
            assert prior_vec.shape == (num_graphs, self.prior_dim)
            prior_vec = prior_vec.to(device)

        if prior_guidance_scale > 0.0:
            if prior_uncond_mode == "zero":
                prior_uncond = torch.zeros((num_graphs, self.prior_dim), device=device)
            elif prior_uncond_mode == "mean":
                if not self.use_prior_norm:
                    raise ValueError("prior_uncond_mode='mean' requires prior_norm stats or provide your own mean.")
               
                prior_uncond = torch.zeros((num_graphs, self.prior_dim), device=device)
            else:
                raise ValueError("prior_uncond_mode must be 'zero' or 'mean'")
        else:
            prior_uncond = None

        pos_traj, v_traj = [], []
        v0_pred_traj, vt_pred_traj = [], []
        ligand_pos, ligand_v = init_ligand_pos, init_ligand_v

        time_seq = list(reversed(range(self.num_timesteps - num_steps, self.num_timesteps)))
        for i in tqdm(time_seq, desc='sampling', total=len(time_seq)):
            t = torch.full(size=(num_graphs,), fill_value=i, dtype=torch.long, device=device)

            # ---- prior CFG ----
            if (prior_guidance_scale > 0.0) and (prior_vec is not None):
                preds_c = self(
                    protein_pos=protein_pos, protein_v=protein_v, batch_protein=batch_protein,
                    init_ligand_pos=ligand_pos, init_ligand_v=ligand_v, batch_ligand=batch_ligand,
                    time_step=t, prior_vec=prior_vec
                )
                preds_u = self(
                    protein_pos=protein_pos, protein_v=protein_v, batch_protein=batch_protein,
                    init_ligand_pos=ligand_pos, init_ligand_v=ligand_v, batch_ligand=batch_ligand,
                    time_step=t, prior_vec=prior_uncond
                )

                if self.model_mean_type == 'noise':
                    eps_c = preds_c['pred_ligand_pos'] - ligand_pos
                    eps_u = preds_u['pred_ligand_pos'] - ligand_pos
                    eps = (1.0 + prior_guidance_scale) * eps_c - prior_guidance_scale * eps_u
                    pos0_from_e = self._predict_x0_from_eps(xt=ligand_pos, eps=eps, t=t, batch=batch_ligand)
                    v0_from_e = (1.0 + prior_guidance_scale) * preds_c['pred_ligand_v'] - prior_guidance_scale * preds_u['pred_ligand_v']
                elif self.model_mean_type == 'C0':
                    
                    x0_c = preds_c['pred_ligand_pos']
                    x0_u = preds_u['pred_ligand_pos']
                    pos0_from_e = (1.0 + prior_guidance_scale) * x0_c - prior_guidance_scale * x0_u
                    v0_from_e = (1.0 + prior_guidance_scale) * preds_c['pred_ligand_v'] - prior_guidance_scale * preds_u['pred_ligand_v']
                else:
                    raise ValueError

            else:
                preds = self(
                    protein_pos=protein_pos, protein_v=protein_v, batch_protein=batch_protein,
                    init_ligand_pos=ligand_pos, init_ligand_v=ligand_v, batch_ligand=batch_ligand,
                    time_step=t, prior_vec=prior_vec
                )
                if self.model_mean_type == 'noise':
                    pred_pos_noise = preds['pred_ligand_pos'] - ligand_pos
                    pos0_from_e = self._predict_x0_from_eps(xt=ligand_pos, eps=pred_pos_noise, t=t, batch=batch_ligand)
                    v0_from_e = preds['pred_ligand_v']
                elif self.model_mean_type == 'C0':
                    pos0_from_e = preds['pred_ligand_pos']
                    v0_from_e = preds['pred_ligand_v']
                else:
                    raise ValueError

            # posterior step
            pos_model_mean = self.q_pos_posterior(x0=pos0_from_e, xt=ligand_pos, t=t, batch=batch_ligand)
            pos_log_variance = extract(self.posterior_logvar, t, batch_ligand)
            nonzero_mask = (1 - (t == 0).float())[batch_ligand].unsqueeze(-1)
            ligand_pos = pos_model_mean + nonzero_mask * (0.5 * pos_log_variance).exp() * torch.randn_like(ligand_pos)

            if not pos_only:
                log_ligand_v_recon = F.log_softmax(v0_from_e, dim=-1)
                log_ligand_v = index_to_log_onehot(ligand_v, self.num_classes)
                log_model_prob = self.q_v_posterior(log_ligand_v_recon, log_ligand_v, t, batch_ligand)
                ligand_v = log_sample_categorical(log_model_prob)

                v0_pred_traj.append(log_ligand_v_recon.clone().cpu())
                vt_pred_traj.append(log_model_prob.clone().cpu())

            ori_ligand_pos = ligand_pos + offset[batch_ligand]
            pos_traj.append(ori_ligand_pos.clone().cpu())
            v_traj.append(ligand_v.clone().cpu())

        ligand_pos = ligand_pos + offset[batch_ligand]
        return {
            'pos': ligand_pos,
            'v': ligand_v,
            'pos_traj': pos_traj,
            'v_traj': v_traj,
            'v0_traj': v0_pred_traj,
            'vt_traj': vt_pred_traj
        }


def extract(coef, t, batch):
    out = coef[t][batch]
    return out.unsqueeze(-1)
