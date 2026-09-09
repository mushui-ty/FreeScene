
from torch import nn
from typing import *
from .networks import *

class MGDiTNetwork(nn.Module):
    def __init__(self,
        node_dim: int, edge_dim: int, # node_dim = 21+1, edge_dim = 10+1
        vqobjfeat_dim: int, objfeat_dim:int, # vqobjfeat_dim= 64+2, objfeat_dim=512(for pool transformer)
        attn_dim=512, t_dim=128,
        global_dim: Optional[int]=None,
        global_condition_dim: Optional[int]=None,
        context_dim: Optional[int]=None,
        n_heads=8, n_layers=5,
        gated_ff=True, dropout=0.1, ada_norm=True,
        cfg_drop_ratio=0.2,
        use_objfeat=True,
        use_d3pm=True,
        use_objfeat_pool_transformer=False,
        use_graphormer=False,
        use_spatial_encoding=False,
        sub = False,
        use_3d_positional_encoding=False,
        not_use_pe=False,
    ):
        super().__init__()
        if not ada_norm:        # not use
            global_dim = t_dim  # if not use AdaLN, use global information in graph-attn instead

        # embed class indices(21+1) to attn_dim
        self.node_embed = nn.Sequential(
            nn.Embedding(node_dim, attn_dim), # node_dim=21+1, attn_dim=512
            nn.GELU(),
            nn.Linear(attn_dim, attn_dim),
        )
        # embed 4*vq_indices(64) to attn_dim
        if use_objfeat and use_d3pm:
            self.vq_indices_embed = nn.Sequential( # 64->128
                nn.Embedding(vqobjfeat_dim, attn_dim//4),
                nn.GELU(),
                nn.Linear(attn_dim//4, attn_dim//4),
            )
        else:
            self.vq_indices_embed = None

        # cat node features and project to attn_dim
        if use_objfeat and not use_d3pm:
            self.node_proj_in = nn.Linear(attn_dim+1280+8, attn_dim)
        elif use_objfeat and use_d3pm:
            self.node_proj_in = nn.Linear(attn_dim+4*attn_dim//4+8, attn_dim)
        else:
            self.node_proj_in = nn.Linear(attn_dim+8, attn_dim)

        # embed edge features to attn_dim//4 = 128
        self.edge_embed = nn.Sequential(
            nn.Embedding(edge_dim, attn_dim//4),
            nn.GELU(),
            nn.Linear(attn_dim//4, attn_dim//4),
        )
        # embed every time in the list to 128 dimensions use sincos_encode
        self.time_embed = nn.Sequential(
            Timestep(t_dim),
            TimestepEmbed(t_dim, t_dim),
        )

        # embed global_text_embeding to t_dim(128)
        if global_condition_dim is not None:
            self.global_condition_embed = nn.Sequential(
                nn.Linear(global_condition_dim,t_dim),
                nn.GELU(),
                nn.Linear(t_dim, t_dim)
            )

        self.transformer_blocks = nn.ModuleList([
            GraphTransformerBlock(
                attn_dim, attn_dim//4, attn_dim, global_dim,
                context_dim, t_dim,
                n_heads, gated_ff, dropout, ada_norm,
                use_graphormer=use_graphormer,
                use_spatial_encoding=use_spatial_encoding,
                sub=sub,
            ) for _ in range(n_layers)
        ])

        if use_spatial_encoding:
            self.box_proj = nn.Sequential(
                nn.Linear(8, attn_dim//2),
                nn.GELU(),
                nn.Linear(attn_dim//2, attn_dim//2),
                nn.GELU(),
            )
            self.extractor = nn.Sequential(
                nn.Linear(attn_dim,attn_dim),
                nn.GELU(),
                nn.Linear(attn_dim, attn_dim)
            )
        self.proj_out_boxes = nn.Sequential(
            nn.LayerNorm(attn_dim),
            nn.Linear(attn_dim, 8)
        )
        self.proj_out_vqobjfeat = nn.Sequential(
            nn.LayerNorm(attn_dim),
            nn.Linear(attn_dim, (vqobjfeat_dim-1)*4) # -1 for mask tocken
        )
        self.proj_out_x = nn.Sequential(
            nn.LayerNorm(attn_dim),
            nn.Linear(attn_dim, node_dim-1) # -1 for mask tocken
        )
        self.proj_out_e = nn.Sequential(
            nn.LayerNorm(attn_dim//4),
            nn.Linear(attn_dim//4, edge_dim-1) # -1 for mask tocken
        )

        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.use_global_info = global_dim is not None
        self.cfg_drop_ratio = cfg_drop_ratio
        self.use_objfeat = use_d3pm
        self.use_d3pm = use_d3pm
        self.use_objfeat_pool_transformer = use_objfeat_pool_transformer
        self.attn_dim = attn_dim
        self.use_spatial_encoding = use_spatial_encoding
        self.use_3d_positional_encoding = use_3d_positional_encoding
        self.not_use_pe = not_use_pe

        if self.use_objfeat_pool_transformer:
            self.objfeat_embed = nn.Embedding(vqobjfeat_dim, objfeat_dim)  # +2 for empty and [mask] token
            self.objfeat_pool_transformer = nn.ModuleList([
                BasicTransformerBlock(  # self-attn + ff
                    objfeat_dim, objfeat_dim, None, None,
                    n_heads, gated_ff, dropout
                ) for _ in range(2)  # TODO: make it configurable
            ])
            self.objfeat_pool_norm = nn.LayerNorm(objfeat_dim)
            self.objfeat_pool = nn.Sequential(
                nn.Linear(objfeat_dim, objfeat_dim),
                nn.GELU(),
                nn.Linear(objfeat_dim, attn_dim)
            )

    def forward(self,
        box: Tensor, x: Tensor, e: Tensor, o: Optional[Tensor],
        t: LongTensor, global_condition: Optional[Tensor]=None,
        condition: Optional[Tensor]=None,
        mask: Optional[LongTensor]=None, condition_mask: Optional[LongTensor]=None,
        cfg_scale=1.
    ):
        # box: (b, n, 8), x: (b, n), e: (b, n, n), o: (b, nk) or (b, n, 1280)
        # 1. process t
        B, N = x.shape[0], x.shape[1]
        if not torch.is_tensor(t):
            if isinstance(t, (int, float)):  # single timestep
                t = torch.tensor([t], device=x.device)
            else:  # list of timesteps
                assert len(t) == x.shape[0]
                t = torch.tensor(t, device=x.device)
        else:  # is tensor
            if t.dim() == 0:
                t = t.unsqueeze(-1).to(x.device)
        t = t * torch.ones(x.shape[0], dtype=t.dtype, device=t.device)# Broadcast to batch dimension, in a way that's campatible with ONNX/Core ML

        # 2. process embedding and concat x
        x_emb = self.node_embed(x) # 1. x embedding (b, n, 512)
        if o is not None and self.use_objfeat and not self.use_d3pm: # 2, o embedding and node projection
            x_emb = self.node_proj_in(torch.cat([x_emb, o, box], dim=-1)) # (b, n, 512)
        elif o is not None and self.use_objfeat and self.use_d3pm:
            if not self.use_objfeat_pool_transformer:
                o = o.reshape(o.shape[0], -1, 4) # (b, n, k) = (128, 12, 4)
                o = self.vq_indices_embed(o) # (b, n, k, 128)
                o = o.reshape(o.shape[0], o.shape[1], -1) # (b, n, 4*128)
                x_emb = self.node_proj_in(torch.cat([x_emb, o, box], dim=-1)) # (b, n, 512)
            if self.use_objfeat_pool_transformer:
                o = o.reshape(B, N, -1)  # (B, N, K)
                o_emb = self.objfeat_embed(o)  # (B, N, K, Do) Do=512
                o_emb = o_emb.reshape(B*N, 4, -1)  # (B*N, K, Do); TODO: make `4` configurable
                o_emb = o_emb + get_1d_sincos_encode(
                    torch.arange(o_emb.shape[1], device=o_emb.device),
                    o_emb.shape[-1], o_emb.shape[1]
                ).unsqueeze(0)
                for block in self.objfeat_pool_transformer:
                    o_emb = block(o_emb)
                o_emb = self.objfeat_pool_norm(o_emb)  # (B*N, K, Do)
                o_emb = torch.mean(o_emb, dim=1).reshape(B, N, -1)  # (B, N, Do); average over objfeat tokens
                o_emb = self.objfeat_pool(o_emb)  # (B, N, attn_dim)
                x_emb = self.node_proj_in(torch.cat([x_emb, o_emb, box], dim=-1)) # (b, n, 512)
        else:
            x_emb = self.node_proj_in(torch.cat([x_emb, box], dim=-1))   # (b, n, 512)
        e_emb = self.edge_embed(e)
        t_emb = self.time_embed(t)
        if self.use_global_info:
            y_emb = t_emb
        else:
            y_emb =None
        if global_condition is not None:
            t_emb += self.global_condition_embed(global_condition)

        # 3. Mask out the diagonal in e_emb
        eye_mask = torch.eye(x.shape[1], device=x.device).bool().unsqueeze(0)  # (1, N, N)
        e_emb = e_emb * (~eye_mask).float().unsqueeze(-1)

        # 4. Positional embeddings
        if self.use_3d_positional_encoding:
            positional_encoding = get_3d_positional_encode(box[:, :, :3].clone(), x_emb.shape[-1])  # (1, n, dx)
        elif self.not_use_pe:
            positional_encoding = 0
        else:
            inst_emb = get_1d_sincos_encode(
                torch.arange(x_emb.shape[1], device=x_emb.device),
                x_emb.shape[-1], x_emb.shape[1]
            ).unsqueeze(0)  # (1, n, dx)
            positional_encoding = inst_emb

        x_emb = x_emb + positional_encoding # may broadcast to (b, n, dx)

        # 5. Spatial encoding (optional)
        spatial_encoding = None
        if self.use_spatial_encoding:
            box_emb = self.box_proj(box) # (b, n, 256)
            spatial_encoding = torch.zeros(B, N, N, self.attn_dim, device=x.device)
            for i in range(N):
                for j in range(N):
                    spatial_encoding[:, i, j, :] = self.extractor(torch.cat([box_emb[:, i, :], box_emb[:, j, :]], dim=-1))

        # Classifier-free guidance dropout during training
        if self.training and self.cfg_drop_ratio > 0.:
            assert cfg_scale == 1., "Do not use `cfg_scale` during training"
            empty_e_emb = torch.zeros_like(e_emb[0]).unsqueeze(0)
            empty_prob = torch.rand(e_emb.shape[0], device=e_emb.device) < self.cfg_drop_ratio
            e_emb[empty_prob, ...] = empty_e_emb

        # Classifier-free guidance during inference (disabled)
        # if not self.training and cfg_scale != 1.:
        #     empty_e_emb = torch.zeros_like(e_emb)
        #     e_emb = torch.cat([empty_e_emb, e_emb], dim=0)
        #     x_emb = torch.cat([x_emb, x_emb], dim=0)
        #     y_emb = torch.cat([y_emb, y_emb], dim=0) if y_emb is not None else None
        #     t_emb = torch.cat([t_emb, t_emb], dim=0)
        #     if condition is not None:
        #         condition = torch.cat([condition, condition], dim=0)
        #     if mask is not None:
        #         mask = torch.cat([mask, mask], dim=0)
        #     if condition_mask is not None:
        #         condition_mask = torch.cat([condition_mask, condition_mask], dim=0)

        for block in self.transformer_blocks:
            x_emb, e_emb, y_emb = block(
                x_emb, e_emb, y_emb, t_emb, condition, mask, condition_mask,
                spatial_encoding=spatial_encoding,
            )
        # x_emb: (b, n, 512), e_emb: (b, n, n, 128)
        out_box = self.proj_out_boxes(x_emb) #(128, 10, 8)
        if mask is not None:
            out_box = out_box * mask.unsqueeze(-1) # (b, 10, 1)
        if self.use_objfeat and self.use_d3pm:
            out_vqobjfeat = self.proj_out_vqobjfeat(x_emb) #(128, 10, 4*65)
            if mask is not None:
                out_vqobjfeat = out_vqobjfeat * mask.unsqueeze(-1)  # (128, 10, 4*65)
            out_vqobjfeat = out_vqobjfeat.reshape(out_vqobjfeat.shape[0], out_vqobjfeat.shape[1], 4, -1) # (128, 10, 4, 65)
            out_objfeat = out_vqobjfeat.reshape(out_vqobjfeat.shape[0], -1, out_vqobjfeat.shape[-1])  # (128, 40, 65)
            out_objfeat = out_objfeat.permute(0, 2, 1) # (128, 65, 40)
        elif self.use_objfeat and not self.use_d3pm:
            pass
        else:
            pass
        out_x = self.proj_out_x(x_emb) #(b, n, 21+1)
        out_x = out_x.permute(0, 2, 1) # (b, 21+1, n)
        out_e = self.proj_out_e(e_emb) #(b, n, n, 10+1)
        out_e = out_e.permute(0, 3, 1, 2) # (b, 10+1, n)
        # # Do classifier-free guidance in inference
        # if not self.training and cfg_scale != 1.:
        #     out_box_uncond, out_box_cond = out_box.chunk(2, dim=0)
        #     out_box = out_box_uncond + cfg_scale * (out_box_cond - out_box_uncond)

        return out_box, out_x, out_e, out_objfeat