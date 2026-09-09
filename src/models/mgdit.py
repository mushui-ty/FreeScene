from typing import *

import numpy as np
from tqdm import tqdm

import torch
def custom_repr(self):
    return f'{{Tensor:{tuple(self.shape)}}} {original_repr(self)}'

original_repr = torch.Tensor.__repr__
torch.Tensor.__repr__ = custom_repr

from torch import nn
from torch import Tensor, LongTensor
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from .mgdit_network import MGDiTNetwork


class MGDiT(nn.Module):
    def __init__(self,
        num_objs:int, num_preds:int,
        num_vqobjfeat_percode=64,
        diffusion_type="ddpm",
        cfg_drop_ratio=0.2,
        use_objfeat=True,
        use_d3pm=True,
        auxiliary_loss_weight=5e-4,
        adaptive_auxiliary_loss=True,
        num_timesteps = 1000,
        text_emb_dim=512,
        use_obj_mask=False,
    ):
        use_objfeat_pool_transformer = False
        use_graphormer = False
        use_spatial_encoding = False
        use_3d_positional_encoding = False
        not_use_pe = False
        sub = False
        super().__init__()
        if diffusion_type == "ddpm":
            self.scheduler = DDPMScheduler(
                num_train_timesteps=num_timesteps,
                beta_start=0.0001, beta_end=0.02,
                beta_schedule="linear",
                variance_type="fixed_small",
                prediction_type="epsilon",
                clip_sample=True,
                clip_sample_range=1.
            )
        elif diffusion_type == "ddim":
            self.scheduler = DDIMScheduler(
                num_train_timesteps=num_timesteps,
                beta_start=0.0001, beta_end=0.02,
                beta_schedule="linear",
                prediction_type="epsilon",
                clip_sample=True,
                clip_sample_range=1.
            )
        else:
            raise NotImplementedError

        self.network = MGDiTNetwork(
            # node_dim=num_objs+1, # 1 for empty
            # edge_dim=num_preds+1, # 1 for empty
            node_dim=num_objs+2, #!----------------------------
            edge_dim=num_preds+2,

            vqobjfeat_dim = num_vqobjfeat_percode+2, objfeat_dim =512,
            t_dim=128, attn_dim=512,
            global_condition_dim= text_emb_dim,
            context_dim= text_emb_dim,
            n_heads=8, n_layers=5,
            gated_ff=True, dropout=0.1, ada_norm=True,
            cfg_drop_ratio=cfg_drop_ratio,
            use_objfeat=use_objfeat,
            use_d3pm=use_d3pm,
            use_objfeat_pool_transformer=use_objfeat_pool_transformer,
            use_graphormer=use_graphormer,
            use_spatial_encoding=use_spatial_encoding,
            sub=sub,
            use_3d_positional_encoding = use_3d_positional_encoding,
            not_use_pe= not_use_pe,
        )
        self.num_objs = num_objs
        self.num_preds = num_preds
        self.num_vqobjfeat_percode = num_vqobjfeat_percode
        self.use_objfeat = use_objfeat
        self.use_d3pm = use_d3pm

        self.cfg_drop_ratio = cfg_drop_ratio
        self.auxiliary_loss_weight = auxiliary_loss_weight
        self.adaptive_auxiliary_loss = adaptive_auxiliary_loss
        self.num_timesteps = num_timesteps
        self.num_node_classes = num_objs+2  # +2 for empty node and [mask] token
        self.num_edge_classes = num_preds+2  # +2 for empty edge and [mask] token
        self.num_objfeat_classes = num_vqobjfeat_percode+2  # +2 for empty and [mask] token;
        self.truncation_rate=1.
        self.use_obj_mask = use_obj_mask

        at, bt, ct, att, btt, ctt = alpha_schedule(num_timesteps, N=num_objs+1)  # +1 for empty node

        at = torch.tensor(at.astype("float64"))
        bt = torch.tensor(bt.astype("float64"))
        ct = torch.tensor(ct.astype("float64"))
        log_at = torch.log(at)
        log_bt = torch.log(bt)
        log_ct = torch.log(ct)
        att = torch.tensor(att.astype("float64"))
        btt = torch.tensor(btt.astype("float64"))
        ctt = torch.tensor(ctt.astype("float64"))
        log_cumprod_at = torch.log(att)
        log_cumprod_bt = torch.log(btt)
        log_cumprod_ct = torch.log(ctt)

        log_1_min_ct = log_1_min_a(log_ct)
        log_1_min_cumprod_ct = log_1_min_a(log_cumprod_ct)

        assert log_add_exp(log_ct, log_1_min_ct).abs().sum().item() < 1e-5
        assert log_add_exp(log_cumprod_ct, log_1_min_cumprod_ct).abs().sum().item() < 1e-5

        # Convert to float32 and register buffers.
        self.register_buffer("log_at", log_at.float())
        self.register_buffer("log_bt", log_bt.float())
        self.register_buffer("log_ct", log_ct.float())
        self.register_buffer("log_cumprod_at", log_cumprod_at.float())
        self.register_buffer("log_cumprod_bt", log_cumprod_bt.float())
        self.register_buffer("log_cumprod_ct", log_cumprod_ct.float())
        self.register_buffer("log_1_min_ct", log_1_min_ct.float())
        self.register_buffer("log_1_min_cumprod_ct", log_1_min_cumprod_ct.float())
        self.zero_vector_x = None
        self.zero_vector_e = None
        self.zero_vector_o = None
        self.predicate_types = ['above', 'left of', 'in front of', 'closely left of', 'closely in front of', 'below', 'right of', 'behind', 'closely right of', 'closely behind']


    def compute_losses(self, sample_params: Dict,
                       text_last_hidden_state: Optional[Tensor]=None,
                       text_embeds: Optional[Tensor]=None,
                       vqvae_model: Optional[nn.Module]=None,
    ) -> Dict[str, Tensor]:
        # input variables
        x = sample_params["objs"]  # (b, n)
        e = sample_params["edges"] #(b, n, n)
        o = sample_params["objfeat_vq_indices"]   # (b, n, k)
        o = o.reshape(x.shape[0], -1)  # (b, n*k)
        mask = sample_params["obj_masks"]  # (b, n)
        if self.use_obj_mask == False:
            mask = None
        boxes = sample_params["boxes"]  # (b, n, 8)
        # batch and device
        B, N, device = x.shape[0], x.shape[1], x.device

        # for classifier-free guidance, set some elements to 0 within a batch in text_emb and global text_emb
        if self.training and self.cfg_drop_ratio > 0.:
            empty_last_hidden_state = torch.zeros_like(text_last_hidden_state[0]).unsqueeze(0)  # e.g., for CLIP ViT-B/32: (1, 77, 512)
            empty_prob = torch.rand(B, device=x.device) < self.cfg_drop_ratio
            text_last_hidden_state[empty_prob, ...] = empty_last_hidden_state # set some text_embed to zero within a batch on a certain probability
            if text_embeds is not None:  # set some gobal text_embed to zero within a batch on a certain probability
                empty_embeds = torch.zeros_like(text_embeds[0]).unsqueeze(0)  # e.g., for CLIP ViT-B/32: (1, 512)
                empty_prob = torch.rand(B, device=x.device) < self.cfg_drop_ratio
                text_embeds[empty_prob, ...] = empty_embeds
        # sample time
        timesteps = torch.randint(1, self.num_timesteps, (B,)).to(device) # self.num_timesteps=1000
        pt = torch.ones_like(timesteps).float()/self.num_timesteps
        # add noise to continuous variables
        noise = torch.randn_like(boxes)
        if mask is not None:
            box_mask = mask.unsqueeze(-1)  # (b, n, 1)
            noise = noise * box_mask
            boxes = boxes * box_mask
        target = noise
        if mask is not None:
            noisy_boxes = self.scheduler.add_noise(boxes, noise, timesteps) * box_mask
        else:
            noisy_boxes = self.scheduler.add_noise(boxes, noise, timesteps)


        # handle whether to add noise to objfeat
        if self.use_objfeat and not self.use_d3pm:# recon objfeat if use_objfeat and not use_d3pm
            pass
            # with torch.no_grad():
            #     # Decode objfeat indices to objfeat embeddings
            #     B, N = o.shape[:2]
            #     o = vqvae_model.reconstruct_from_indices(
            #         o.reshape(B*N, -1)
            #     ).reshape(B, N, -1)
        elif self.use_objfeat and self.use_d3pm: # add noise to objfeat_vq_indices if use_objfeat and use_d3pm
            x_start = x #! -------------------------------------------------------------------------
            log_x_start = index_to_log_onehot(x_start, self.num_node_classes) # (b, 21+2, n)
            log_xt = self.q_sample(log_x_start=log_x_start, t=timesteps, num_classes=self.num_node_classes) # (b, 21+2, n) forward process sampling
            x = log_onehot_to_index(log_xt) #! (b, n)
            e_start = e.reshape(B, -1) # (b, n*n)
            log_e_start = index_to_log_onehot(e_start, self.num_edge_classes) # (b, 10+2, n*n)
            log_et = self.q_sample(log_x_start=log_e_start, t=timesteps, num_classes=self.num_edge_classes) # (b, 10+2, n*n) forward process sampling
            e = log_onehot_to_index(log_et).reshape(B, N, N) #! (b, n, n)
            o_start = o
            log_o_start = index_to_log_onehot(o_start, self.num_objfeat_classes)# (B, Co, NK)=(128, 66, 40)
            log_ot = self.q_sample(log_x_start=log_o_start, t=timesteps, num_classes=self.num_objfeat_classes) #(128, 66, 40) forward process sampling
            o = log_onehot_to_index(log_ot) #!(b, n*k)
        else:
            o = None


        # network forward for prediction noise, box_pred: (b, n, 8) out_x:(b, 21+1, n) out_e:(b, 10+1, n, n)    out_o:(b, 4n) -> (B, 64+1, nK)
        box_pred, out_x, out_e, out_o = self.network(
            noisy_boxes, x, e, o, timesteps, text_embeds, text_last_hidden_state, mask=mask
        )

        # compute continuous boxes loss
        losses = {}
        k = 1.0
        for i in x.shape:
            k *= i
        losses["pos_mse"] = F.mse_loss(box_pred[..., :3], target[..., :3], reduction="none").sum() / k
        losses["size_mse"] = F.mse_loss(box_pred[..., 3:6], target[..., 3:6], reduction="none").sum() / k
        losses["angle_mse"] = F.mse_loss(box_pred[..., 6:8], target[..., 6:8], reduction="none").sum() / k

        # compute discrete loss
        if self.use_objfeat and self.use_d3pm:
            #!-------------------------
            #softmax
            log_pred_x = F.log_softmax(out_x.double(), dim=1).float() #(b, 21+1, n)
            log_pred_x = self.truncate(log_pred_x, self.truncation_rate)
            if self.zero_vector_x is None or self.zero_vector_x.shape[0] != B or self.zero_vector_x.shape[2:] != log_pred_x.shape[2:]:
                self.zero_vector_x = torch.zeros(B, 1, *log_pred_x.shape[2:]).type_as(out_x) - 70.
            log_pred_x = torch.cat([log_pred_x, self.zero_vector_x], dim=1)  # (b, 21+2, n)
            log_x_recon = torch.clamp(log_pred_x, -70., 0.) #(b, 21+2, n)

            log_pred_e = F.log_softmax(out_e.double(), dim=1).float() #(b, 10+1, n, n)
            log_pred_e = self.truncate(log_pred_e, self.truncation_rate)
            if self.zero_vector_e is None or self.zero_vector_e.shape[0] != B or self.zero_vector_e.shape[2:] != log_pred_e.shape[2:]:
                self.zero_vector_e = torch.zeros(B, 1, *log_pred_e.shape[2:]).type_as(out_e) - 70.
            log_pred_e = torch.cat([log_pred_e, self.zero_vector_e], dim=1)  # (b, 10+2, n, n)
            log_e_recon = torch.clamp(log_pred_e, -70., 0.) #(b, 10+2, n, n)

            log_pred_o = F.log_softmax(out_o.double(), dim=1).float() #(B, Co, NK)
            log_pred_o = self.truncate(log_pred_o, self.truncation_rate)
            if self.zero_vector_o is None or self.zero_vector_o.shape[0] != B or self.zero_vector_o.shape[2:] != log_pred_o.shape[2:]:
                self.zero_vector_o = torch.zeros(B, 1, *out_o.shape[2:]).type_as(out_o) - 70.
            log_pred_o = torch.cat([log_pred_o, self.zero_vector_o], dim=1)  # (B, Co+1, NK)
            log_o_recon = torch.clamp(log_pred_o, -70., 0.) #(B, Co, NK)->(128, 66, 48) (1 for mask and 1 for zero_vector)

            # compute q(xt_1|xt,x0) using log_o_recon
            log_model_prob_x = self.q_posterior(log_x_start=log_x_recon, log_x_t=log_xt, t=timesteps, num_classes=self.num_node_classes)  # (b, 21+2, n)
            log_model_prob_e = self.q_posterior(log_x_start=log_e_recon.reshape(B, -1, N*N), log_x_t=log_et, t=timesteps, num_classes=self.num_edge_classes) #(b, 10+2, n*n)
            log_model_prob_o = self.q_posterior(log_x_start=log_o_recon, log_x_t=log_ot, t=timesteps, num_classes=self.num_objfeat_classes) # (b, 64+2, nK)
            # compute q(xt|xt_1,x0) using true log_o_start
            log_true_prob_x = self.q_posterior(log_x_start=log_x_start, log_x_t=log_xt, t=timesteps, num_classes=self.num_node_classes)  # (b, 21+2, n)
            log_true_prob_e = self.q_posterior(log_x_start=log_e_start, log_x_t=log_et, t=timesteps, num_classes=self.num_edge_classes) #(b, 10+2, n*n)
            log_true_prob_o = self.q_posterior(log_x_start=log_o_start, log_x_t=log_ot, t=timesteps, num_classes=self.num_objfeat_classes) # (b, 64+2, nK)
            kl_x = self.multinomial_kl(log_true_prob_x, log_model_prob_x)  # (b, n)
            kl_e = self.multinomial_kl(log_true_prob_e, log_model_prob_e)  # (b, n*n)
            kl_o = self.multinomial_kl(log_true_prob_o, log_model_prob_o) #(b, nK)

            decoder_nll_x = -log_categorical(log_x_start, log_model_prob_x)  # (b, n)
            t0_mask_x = (timesteps== 0).unsqueeze(1).repeat(1, log_x_start.shape[-1])  # (b, n)
            kl_loss_x = torch.where(t0_mask_x, decoder_nll_x, kl_x)  # (b, n): kl_loss = decoder_nll when t==0 else kl_x
            discrete_loss_x = kl_loss_x.mean(dim=1).sum()  # (1,)

            decoder_nll_e = -log_categorical(log_e_start, log_model_prob_e)  # (b, n*n)
            t0_mask_e = (timesteps== 0).unsqueeze(1).repeat(1, log_e_start.shape[-1])  # (b, n*n)
            kl_loss_e = torch.where(t0_mask_e, decoder_nll_e, kl_e)  # (b, n*n): kl_loss = decoder_nll when t==0 else kl_e
            discrete_loss_e = kl_loss_e.mean(dim=1).sum()  # (1,)

            decoder_nll_o = -log_categorical(log_o_start, log_model_prob_o)  # (b, nK)
            t0_mask_o = (timesteps== 0).unsqueeze(1).repeat(1, log_o_start.shape[-1])  # (b, nK)
            kl_loss_o = torch.where(t0_mask_o, decoder_nll_o, kl_o)  # (b, nK): kl_loss = decoder_nll when t==0 else kl_o
            discrete_loss_o = kl_loss_o.mean(dim=1).sum()  # (1,)
            # compute auxiliary loss
            if self.auxiliary_loss_weight != 0:
                kl_aux_x = self.multinomial_kl(log_x_start[:, :-1, ...], log_x_recon[:, :-1, ...])  # (b, n)
                kl_aux_e = self.multinomial_kl(log_e_start[:, :-1, ...], log_e_recon.reshape(B, -1, N*N)[:, :-1, ...])  # (b, n*n)
                kl_aux_o = self.multinomial_kl(log_o_start[:, :-1, ...], log_o_recon[:, :-1, ...])  # (b, nK)
                if self.adaptive_auxiliary_loss:
                    addition_loss_weight = (1. - timesteps/self.num_timesteps) + 1.
                else:
                    addition_loss_weight = 1.
                kl_aux_loss_x = torch.where(t0_mask_x, decoder_nll_x, kl_aux_x)  # (b, n): loss2 = decoder_nll when t==0 else kl_aux_x
                kl_aux_loss_x = kl_aux_loss_x.mean(dim=1)  # (b,)
                kl_aux_loss_x =  addition_loss_weight * kl_aux_loss_x * self.auxiliary_loss_weight
                discrete_loss_x += kl_aux_loss_x.sum()  # (1,)

                kl_aux_loss_e = torch.where(t0_mask_e, decoder_nll_e, kl_aux_e)  # (b, n*n): loss2 = decoder_nll when t==0 else kl_aux_e
                kl_aux_loss_e = kl_aux_loss_e.mean(dim=1)  # (b,)
                kl_aux_loss_e =  addition_loss_weight * kl_aux_loss_e * self.auxiliary_loss_weight
                discrete_loss_e += kl_aux_loss_e.sum()  # (1,)

                kl_aux_loss_o = torch.where(t0_mask_o, decoder_nll_o, kl_aux_o)  # (b, nK): loss2 = decoder_nll when t==0 else kl_aux_o
                kl_aux_loss_o = kl_aux_loss_o.mean(dim=1)  # (b,)
                kl_aux_loss_o =  addition_loss_weight * kl_aux_loss_o * self.auxiliary_loss_weight
                discrete_loss_o += kl_aux_loss_o.sum()  # (1,)
            losses["discrete_loss_x"] = discrete_loss_x
            losses["discrete_loss_e"] = discrete_loss_e
            losses["discrete_loss_o"] = discrete_loss_o
        elif self.use_objfeat and not self.use_d3pm:
            pass
        else:
            pass
        return losses
    @torch.no_grad()
    def generate_samples(self,
        batch_size: int, num_nodes: int,
        text_last_hidden_state: Optional[Tensor]=None, text_embeds: Optional[Tensor]=None,
        num_timesteps: Optional[int]=100,
        cfg_scale=1.,
        vqvae_model=None,
        x = None, e = None, o = None, # x: (b, n) e: (b, n, n) o: (b, n, k)
        rearrange=False,
        boxes_fixed=None,
        full_step=True,
        not_complete=False,
        num_diffusion_steps=1000,
    ):
        if full_step:
            num_timesteps = num_diffusion_steps
        # when x is None, generate from scratch
        # when given x and e, but o is None , generate or complete on graph(stylization simuteanously)
        # when given x, o and boxes, but e is None and N=n , rearrange #!rearrange=True and boxes_fixed is not None
        # when given x, e and o, generate on graph and have to use our retrived base stylization in advance
        # when given x, e and boxes, stylize o #! rearrange=False and boxes_fixed is not None
        self.cfg_scale = cfg_scale
        B, N = batch_size, num_nodes
        device = text_last_hidden_state.device
        # fixed conditions mask
        if x is not None:
            assert x.shape[0] == B and x.shape[1] <= N
            n = x.shape[1]
            x_flag = torch.zeros((B, N), device=device).long()
            x = F.pad(x, (0, N-n, 0, 0), mode='constant', value=self.num_objs).long()
            if e is not None:
                assert e.shape[0] == B and e.shape[1] == x.shape[1] and e.shape[2] == x.shape[1]
                e_flag = torch.zeros((B, N, N), device=device).long()
                e = F.pad(e, (0, N-n, 0, N-n, 0, 0), mode='constant', value=self.num_preds).long()
                # compute reverse edges
                for b in range(B):
                    for i in range(n):
                        for j in range(n):
                            if e[b, i, j] != self.num_preds and e[b, i, j] != self.num_preds+1:
                                e[b, j, i] = self.predicate_types.index(
                                    reverse_rel(self.predicate_types[e[b, i, j]])
                                )
            if o is not None:
                assert o.shape[0] == B and o.shape[1] == x.shape[1]
                o_flag = torch.zeros((B, N, o.shape[2]), device=device).long()
                o = F.pad(o, (0, 0, 0, N-n, 0, 0), mode='constant', value=self.num_vqobjfeat_percode).long()
            # flag the mask for unchangeable condition
            for b in range(B):
                for i in range(n):
                    if x[b, i] != self.num_objs:
                        x_flag[b, i] = 1
                    if o is not None:
                        if not o[b, i, :].equal(torch.full((o.shape[2],),  self.num_vqobjfeat_percode, device=device).long()):
                            o_flag[b, i, :] = 1
                    if e is not None:
                        for j in range(n):
                            if e[b, i, j] != self.num_preds:
                                e_flag[b, i, j] = 1
        # prepare noise(all-masked graphs)
        boxes = torch.randn(B, N, 8).to(device) # prepare continous boxes noise
        if rearrange: #B= 1, N = n
            assert boxes_fixed is not None
            box_flag = x_flag.unsqueeze(2).repeat(1, 1, 8)
            box_flag[:, :, :3] = 0
            box_flag[:, :, 6:8] = 0
            box_flag = box_flag.bool() # (b, n, 8)
        elif boxes_fixed is not None:
            box_flag = x_flag.unsqueeze(2).repeat(1, 1, 8)
            box_flag = box_flag.bool() # (b, n, 8)
        # zero_logits_x = torch.zeros((B, self.num_node_classes-1, N), device=device)  #(b, 21+1, n)
        # one_logits_x = torch.ones((B, 1, N), device=device)  #(b, 1, n)
        # mask_logits_x = torch.cat([zero_logits_x, one_logits_x], dim=1)  #(b, 21+2, n)
        # log_z_x_mask = torch.log(mask_logits_x)  #!(b, 21+2, n)
        z_x_mask = torch.full((B, N), self.num_node_classes-1, device=device).long()  # (b, n)
        z_x_empty = torch.full((B, N), self.num_node_classes-2, device=device).long()  # (b, n)

        # zero_logits_e = torch.zeros((B, self.num_edge_classes-1, N*N), device=device)  #(b, 10+1, n*n)
        # one_logits_e = torch.ones((B, 1, N*N), device=device)  #(b, 1, n*n)
        # mask_logits_e = torch.cat([zero_logits_e, one_logits_e], dim=1)  #(b, 10+2, n*n)
        # log_z_e_mask = torch.log(mask_logits_e)  #(b, 10+2, n*n)
        # log_z_e_mask = log_z_e_mask.reshape(B, -1, N, N)  #!(b, 10+2, n, n)
        z_e_mask = torch.full((B, N, N), self.num_edge_classes-1, device=device).long()  # (b, n, n)
        z_e_empty = torch.full((B, N, N), self.num_edge_classes-2, device=device).long()  # (b, n, n)

        # zero_logits_o = torch.zeros((B, self.num_objfeat_classes-1, N*4), device=device)
        # one_logits_o = torch.ones((B, 1, N*4),device=device)
        # mask_logits_o = torch.cat([zero_logits_o, one_logits_o], dim=1) #(b, 64+2, n*4)
        # log_z_o_mask = torch.log(mask_logits_o)  #!(b, 64+2, n*4)
        z_o_mask = torch.full((B, N*4), self.num_objfeat_classes-1, device=device).long()  # (b, n, k)
        z_o_mask = z_o_mask.reshape(B, -1) # (b, n*k)
        z_o_empty = torch.full((B, N*4), self.num_objfeat_classes-2, device=device).long()  # (b, n, k)
        z_o_empty = z_o_empty.reshape(B, -1) # (b, n*k)

        # prepare clear graphs and fixed condition flag
        if x is not None:
            fixed_x = x #(b, n)
            x_flag = x_flag.bool() #(b, n)
            if e is not None:
                fixed_e = e #(b, n, n)
                e_flag = e_flag.bool() #(b, n, n)
            if o is not None:
                fixed_o = o.reshape(B, -1) #(b, n*k)
                o_flag = o_flag.bool().reshape(B, -1) #(b, n*k)
        # iterative sampling
        x_in = z_x_mask # (b, n)
        e_in = z_e_mask # (b, n, n)
        o_in = z_o_mask # (b, n*k)
        self.scheduler.set_timesteps(num_timesteps) #num_timesteps=100
        for diffusion_index in tqdm(range(num_timesteps-1, -1, -1), desc="Generating scenes", ncols=125):
            if full_step:
                t = torch.full((B,), diffusion_index, device=device, dtype=torch.long)
            else:
                t = torch.full((B,), diffusion_index * 10, device=device, dtype=torch.long)
            if x is not None:
                if not_complete:
                    x_in = torch.where(x_flag, fixed_x, z_x_empty)
                else:
                    x_in = torch.where(x_flag, fixed_x, x_in)
            if e is not None:
                if not_complete:
                    e_in = torch.where(e_flag, fixed_e, z_e_empty)
                else:
                    e_in = torch.where(e_flag, fixed_e, e_in)
            if o is not None:
                if not_complete:
                    o_in = torch.where(o_flag, fixed_o, z_o_empty)
                else:
                    o_in = torch.where(o_flag, fixed_o, o_in)
            if rearrange:
                boxes = torch.where(box_flag, boxes_fixed, boxes) # (b, n, 8)
            elif boxes_fixed is not None:
                boxes = torch.where(box_flag, boxes_fixed, boxes) # (b, n, 8)
            boxes, log_z_x, log_z_e, log_z_o = self.mix_p_sample(
                boxes, x_in, e_in, o_in, t, text_embeds, text_last_hidden_state,
            )
            x_in = log_onehot_to_index(log_z_x)
            e_in = log_onehot_to_index(log_z_e)
            o_in = log_onehot_to_index(log_z_o)

        if x is not None:
            x_out = torch.where(x_flag, fixed_x, x_in) # (b, n)
        else:
            x_out = x_in # (b, n)
        if e is not None:
            e_out = torch.where(e_flag, fixed_e, e_in) # (b, n, n)
        else:
            e_out = e_in # (b, n, n)
        if o is not None:
            o_out = torch.where(o_flag, fixed_o, o_in).reshape(B, N, -1) # (b, n, k)
        else:
            o_out = o_in.reshape(B, N, -1) # (b, n, k)
        if rearrange:
            boxes_out = torch.where(box_flag, boxes_fixed, boxes) # (b, n, 8)
        elif boxes_fixed is not None:
            boxes_out = torch.where(box_flag, boxes_fixed, boxes) # (b, n, 8)
        else:
            boxes_out = boxes # (b, n, 8)
        return boxes_out, x_out, e_out, o_out # boxes_out: (B, n, 8) x_out: (B, n) e_out: (B, n, n) o_out: (B, n, k)



    def q_posterior(self,
        log_x_start: Tensor, log_x_t: Tensor, t: LongTensor,
        num_classes: int
    ):  # p_theta(xt_1|xt) = sum( q(xt-1|xt,x0') * p(x0') )
        assert t.min().item() >= 0 and t.max().item() < self.num_timesteps

        batch_size = log_x_start.shape[0]
        onehot_x_t = log_onehot_to_index(log_x_t)
        mask = (onehot_x_t == num_classes-1).unsqueeze(1)  # -1 for [mask] token
        log_one_vector = torch.zeros(batch_size, 1, *([1] * (len(log_x_start.shape)-2))).type_as(log_x_t)
        log_zero_vector = torch.log(log_one_vector+1e-30).expand(-1, -1, *log_x_start.shape[2:])

        log_qt = self.q_pred(log_x_t, t, num_classes)  # q(xt|x0)
        log_qt = log_qt[:, :-1, ...]
        log_cumprod_ct = extract(self.log_cumprod_ct, t, log_x_start.shape)  # ct~
        ct_cumprod_vector = log_cumprod_ct.expand(-1, num_classes-1, *([-1]*(len(log_x_start.shape)-2)))
        log_qt = (~mask)*log_qt + mask*ct_cumprod_vector

        log_qt_one_timestep = self.q_pred_one_timestep(log_x_t, t, num_classes)  # q(xt|xt_1)
        log_qt_one_timestep = torch.cat([log_qt_one_timestep[:, :-1, ...], log_zero_vector], dim=1)
        log_ct = extract(self.log_ct, t, log_x_start.shape)  # ct
        ct_vector = log_ct.expand(-1, num_classes-1, *([-1]*(len(log_x_start.shape)-2)))
        ct_vector = torch.cat([ct_vector, log_one_vector], dim=1)
        log_qt_one_timestep = (~mask)*log_qt_one_timestep + mask*ct_vector

        q = log_x_start[:, :-1, ...] - log_qt
        q = torch.cat([q, log_zero_vector], dim=1)
        q_log_sum_exp = torch.logsumexp(q, dim=1, keepdim=True)
        q = q - q_log_sum_exp
        log_EV_xtmin_given_xt_given_xstart = self.q_pred(q, t-1, num_classes) + log_qt_one_timestep + q_log_sum_exp

        return torch.clamp(log_EV_xtmin_given_xt_given_xstart, -70., 0.)
    def q_pred_one_timestep(self, log_x_t: Tensor, t: LongTensor, num_classes: int):  # q(xt|xt_1)
        log_at = extract(self.log_at, t, log_x_t.shape)  # at
        log_bt = extract(
            (self.log_bt.exp() * (self.num_node_classes-1) / (num_classes-1)).log(),
            t, log_x_t.shape
        )  # bt; -1 for [mask] token
        log_ct = extract(self.log_ct, t, log_x_t.shape)  # ct
        log_1_min_ct = extract(self.log_1_min_ct, t, log_x_t.shape)  # 1-ct

        _prob_sum = log_at.exp() + log_bt.exp() * (num_classes-1) + log_ct.exp()
        assert torch.allclose(_prob_sum, torch.ones_like(_prob_sum))

        log_probs = torch.cat(
            [
                log_add_exp(log_x_t[:, :-1, ...] + log_at, log_bt),
                log_add_exp(log_x_t[:, -1:, ...] + log_1_min_ct, log_ct)
            ],
            dim=1
        )

        return log_probs
    def q_sample(self, log_x_start: Tensor, t: LongTensor, num_classes: int):  # discrete forward process, q(xt|x0) and sample xt
        log_EV_qxt_x0 = self.q_pred(log_x_start, t, num_classes)

        # Gumbel sample
        log_sample = self.log_sample_categorical(log_EV_qxt_x0, num_classes)
        return log_sample

    def q_pred(self, log_x_start: Tensor, t: LongTensor, num_classes: int):  # q(xt|x0) for forward process
        t = (t + (self.num_timesteps + 1))%(self.num_timesteps + 1)
        log_cumprod_at = extract(self.log_cumprod_at, t, log_x_start.shape)  # at~
        log_cumprod_bt = extract(
            (self.log_cumprod_bt.exp() * (self.num_node_classes-1) / (num_classes-1)).log(),
            t, log_x_start.shape
        )  # bt~; -1 for [mask] token
        log_cumprod_ct = extract(self.log_cumprod_ct, t, log_x_start.shape)  # ct~
        log_1_min_cumprod_ct = extract(self.log_1_min_cumprod_ct, t, log_x_start.shape)  # 1-ct~

        _prob_sum = log_cumprod_at.exp() + log_cumprod_bt.exp() * (num_classes-1) + log_cumprod_ct.exp()
        assert torch.allclose(_prob_sum, torch.ones_like(_prob_sum))

        log_probs = torch.cat(
            [
                log_add_exp(log_x_start[:, :-1, ...] + log_cumprod_at, log_cumprod_bt),
                log_add_exp(log_x_start[:, -1:, ...] + log_1_min_cumprod_ct, log_cumprod_ct)
            ],
            dim=1
        )

        return log_probs
    def mix_p_sample(
        self, boxes, x_in, e_in, o_in, t, text_embeds, text_last_hidden_state,
        cfg_scale=1.,
    ):
        # x_in: (b, n), e_in: (b, n, n), o_in: (b, n*k)
        # get the log_onehot of x_in, e_in, o_in
        log_onehot_x = index_to_log_onehot(x_in, self.num_node_classes)
        log_onehot_e = index_to_log_onehot(e_in, self.num_edge_classes)
        log_onehot_o = index_to_log_onehot(o_in, self.num_objfeat_classes)
        # 1. sample start i.e.p(x0|xt) for predicting discrete q(xt_1|xt,x0) , at the same, get noise for continuous diffusion
        # prepare variables for classifier-free guidance
        if cfg_scale != 1.:
            empty_cond_emb = torch.zeros_like(text_last_hidden_state)  # e.g., for CLIP ViT-B/32: (B, 77, 512)
            text_last_hidden_state = torch.cat([empty_cond_emb, text_last_hidden_state], dim=0)
            boxes = torch.cat([boxes, boxes], dim=0)
            x_in = torch.cat([x_in, x_in], dim=0)
            e_in = torch.cat([e_in, e_in], dim=0)
            o_in = torch.cat([o_in, o_in], dim=0)
            t = torch.cat([t, t], dim=0)
            if text_embeds is not None:
                empty_global_cond_emb = torch.zeros_like(text_embeds)  # e.g., for CLIP ViT-B/32: (B, 512)
                text_embeds = torch.cat([empty_global_cond_emb, text_embeds], dim=0)
        # o: (b, 66, 4n) -> (b, 65, 4n)
        #! network forward process
        box_pred, out_x, out_e, out_o = self.network(
            boxes, x_in, e_in, o_in, t, text_embeds, text_last_hidden_state
        )
        log_pred_o = F.log_softmax(out_o.double(), dim=1).float() #(b, 65, 4n)
        log_pred_x = F.log_softmax(out_x.double(), dim=1).float() #(b, 21+1, n)
        log_pred_e = F.log_softmax(out_e.double(), dim=1).float() #(b, 10+1, n, n)
        batch_size = log_pred_o.shape[0]
        # classifier-free guiance noise computing for cfg_scale != 1.
        if cfg_scale != 1.:
            log_pred_x_uncond, log_pred_x_cond = log_pred_x.chunk(2, dim=0)
            log_pred_x = log_pred_x_uncond + cfg_scale * (log_pred_x_cond - log_pred_x_uncond)
            log_pred_x -= torch.logsumexp(log_pred_x, dim=1, keepdim=True)  # normalize
            log_pred_e_uncond, log_pred_e_cond = log_pred_e.chunk(2, dim=0)
            log_pred_e = log_pred_e_uncond + cfg_scale * (log_pred_e_cond - log_pred_e_uncond)
            log_pred_e -= torch.logsumexp(log_pred_e, dim=1, keepdim=True)  # normalize
            log_pred_o_uncond, log_pred_o_cond = log_pred_o.chunk(2, dim=0)
            log_pred_o = log_pred_o_uncond + cfg_scale * (log_pred_o_cond - log_pred_o_uncond)
            log_pred_o -= torch.logsumexp(log_pred_o, dim=1, keepdim=True)  # normalize
            t, cond_emb = t[batch_size:, ...], cond_emb[batch_size:, ...]
            if global_cond_emb is not None:
                global_cond_emb = global_cond_emb[batch_size:, ...]
            out_box_uncond, out_box_cond = box_pred.chunk(2, dim=0)
            box_pred = out_box_uncond + cfg_scale * (out_box_cond - out_box_uncond)
        # compute log_start
        log_pred_x = self.truncate(log_pred_x, self.truncation_rate)
        self.zero_vector_x = torch.zeros(batch_size, 1, *log_pred_x.shape[2:]).type_as(out_x) - 70.
        log_pred_x = torch.cat([log_pred_x, self.zero_vector_x], dim=1)
        log_x_recon = torch.clamp(log_pred_x, -70., 0.) #(B, 21+2, n)-> (1 for mask and 1 for zero_vector)
        log_pred_e = self.truncate(log_pred_e, self.truncation_rate)
        self.zero_vector_e = torch.zeros(batch_size, 1, *log_pred_e.shape[2:]).type_as(out_e) - 70.
        log_pred_e = torch.cat([log_pred_e, self.zero_vector_e], dim=1)
        log_e_recon = torch.clamp(log_pred_e, -70., 0.) #(B, 10+2, n, n)-> (1 for mask and 1 for zero_vector)
        log_pred_o = self.truncate(log_pred_o, self.truncation_rate)
        self.zero_vector_o = torch.zeros(batch_size, 1, *log_pred_o.shape[2:]).type_as(out_o) - 70.
        log_pred_o = torch.cat([log_pred_o, self.zero_vector_o], dim=1)
        log_o_recon = torch.clamp(log_pred_o, -70., 0.) #(B, 66, 4n)-> (1 for mask and 1 for zero_vector)


        #2. we have get the probability distribution p(x0|xt) given xt,
        # now we use it to compute q(xt_1|xt,x0)=sum( q(xt-1|xt,x0) * p(x0|xt) )
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        log_model_pred_x = self.q_posterior(
                log_x_start=log_x_recon, log_x_t=log_onehot_x, t=t,
                num_classes=self.num_node_classes
        ) #(B, 21+2, n)
        log_model_pred_e = self.q_posterior(
                log_x_start=log_e_recon, log_x_t=log_onehot_e, t=t,
                num_classes=self.num_edge_classes
        ) #(B, 10+2, n, n)
        log_model_pred_o = self.q_posterior(
                log_x_start=log_o_recon, log_x_t=log_onehot_o, t=t,
                num_classes=self.num_objfeat_classes
        ) #(B, 64+2, 4n)

        #3. Gumbel sample xt-1 using q(xt_1|xt,x0)
        out_x = self.log_sample_categorical(log_model_pred_x, self.num_node_classes) #(B, 21+2, n)
        out_e = self.log_sample_categorical(log_model_pred_e, self.num_edge_classes) #(B, 10+2, n, n)
        out_o = self.log_sample_categorical(log_model_pred_o, self.num_objfeat_classes) #(B, 64+2, 4n)

        #4. continuous sampling
        out_boxes = self.scheduler.step(box_pred, t[0], boxes).prev_sample

        return out_boxes, out_x, out_e, out_o




    def log_sample_categorical(self, logits: Tensor, num_classes: int):  # use gumbel to sample onehot vector from log probability
        uniform = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
        sample = (gumbel_noise + logits).argmax(dim=1)

        log_sample = index_to_log_onehot(sample, num_classes)
        return log_sample
    def truncate(self, log_p_x_0: Tensor, truncation_rate=1.):
        sorted_log_p_x_0, indices = torch.sort(log_p_x_0, dim=1, descending=True)
        sorted_p_x_0 = torch.exp(sorted_log_p_x_0)
        keep_mask = sorted_p_x_0.cumsum(dim=1) < truncation_rate

        # Ensure that at least the largest probability is not zeroed out
        all_true = torch.full_like(keep_mask[:, 0:1, :], True)
        keep_mask = torch.cat((all_true, keep_mask), dim=1)
        keep_mask = keep_mask[:, :-1, :]

        keep_mask = keep_mask.gather(1, indices.argsort(1))

        rv = log_p_x_0.clone()
        rv[~keep_mask] = -torch.inf  # -inf = log(0)
        return rv
    def multinomial_kl(self, log_prob1: Tensor, log_prob2: Tensor):  # compute KL loss on log_prob
        kl = (log_prob1.exp() * (log_prob1 - log_prob2)).sum(dim=1)
        return kl

def index_to_log_onehot(x: LongTensor, num_classes: int):
    assert x.max().item() < num_classes, f"Error: {x.max().item()} >= {num_classes}"
    # x(B,12), x_onehot(B,12,23)
    x_onehot: Tensor = F.one_hot(x, num_classes)
    permute_order = (0, -1) + tuple(range(1, len(x.shape))) # (0,-1,1)
    x_onehot = x_onehot.permute(permute_order) # (b,23,12)
    log_x = torch.log(x_onehot.float().clamp(min=1e-30))
    return log_x

def extract(a: Tensor, t: LongTensor, x_shape: Tuple[int, ...]):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def log_add_exp(a: Tensor, b: Tensor):
    maximum = torch.max(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))

def log_onehot_to_index(log_x: Tensor):
    return log_x.argmax(dim=1)

def log_categorical(log_x_start: Tensor, log_prob: Tensor):
    return (log_x_start.exp() * log_prob).sum(dim=1)

def alpha_schedule(time_step: int, N: int, att_1=0.99999, att_T=0.000009, ctt_1=0.000009, ctt_T=0.99999):
    att = np.arange(0, time_step) / (time_step-1) * (att_T - att_1) + att_1
    att = np.concatenate([[1], att])
    at = att[1:] / att[:-1]

    ctt = np.arange(0, time_step) / (time_step-1) * (ctt_T - ctt_1) + ctt_1
    ctt = np.concatenate([[0], ctt])
    one_minus_ctt = 1. - ctt
    one_minus_ct = one_minus_ctt[1:] / one_minus_ctt[:-1]
    ct = 1. - one_minus_ct

    bt = (1. - at - ct) / N

    att = np.concatenate([att[1:], [1]])
    ctt = np.concatenate([ctt[1:], [0]])
    btt = (1. - att - ctt) / N

    return at, bt, ct, att, btt, ctt

def log_1_min_a(a: Tensor):
    return torch.log(1. - a.exp() + 1e-40)

def reverse_rel(rel: str) -> str:
    return {
        "above": "below",
        "below": "above",
        "in front of": "behind",
        "behind": "in front of",
        "left of": "right of",
        "right of": "left of",
        "closely in front of": "closely behind",
        "closely behind": "closely in front of",
        "closely left of": "closely right of",
        "closely right of": "closely left of"
    }[rel]
