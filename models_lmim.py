from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from vits import VIT, Patchify, PatchEmbed
from xvits import CrossDecoder


def _build_mlp(in_dim, out_dim, hidden_dim=4096, num_layers=1, norm=nn.LayerNorm, out_norm=False):
    if num_layers == 0:
        return None
    projector = nn.Sequential()
    for l in range(num_layers):
        dim1 = in_dim if l == 0 else hidden_dim
        dim2 = out_dim if l == num_layers - 1 else hidden_dim
        projector.add_module(f"linear{l}", nn.Linear(dim1, dim2, bias=False))
        if out_norm or l < num_layers - 1:
            projector.add_module(f"norm{l}", norm(dim2))
        if l < num_layers - 1:
            projector.add_module(f"act{l}", nn.GELU())
    return projector


class LMIM(nn.Module):
    """ Latent Masked Image Modeling with VisionTransformer backbone
    """
    def __init__(self, grid_size=14, patch_size=16, patch_gap=0, in_chans=3,
                 embed_dim=1024, num_heads=16, depth=24, target_depth=0, 
                 decoder_embed_dim=512, decoder_depth=3, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm,
                 drop=0., attn_drop=0., drop_path=0.,
                 tau=0.2, num_vis=20, avg_vis_mask_token=True,
                 avg_sim_coeff=0., mask_target=True,
                 loss='infonce_patches', freeze_pe=None, proj_cfg=None,
                 use_hr_gram_loss=False, hr_gram_loss_type="gram", sr_scale_factor=2):
        super().__init__()

        self.loss = loss
        self.patch_gap = patch_gap
        self.tau = tau
        self.num_vis = num_vis
        self.avg_sim_coeff = avg_sim_coeff
        self.use_hr_gram_loss = use_hr_gram_loss
        self.hr_gram_loss_type = hr_gram_loss_type
        self.sr_scale_factor = sr_scale_factor
        self.grid_size = grid_size

        self.patchify = Patchify(patch_size=patch_size, grid_size=grid_size, in_chans=in_chans)
        self.hr_patchify = Patchify(patch_size=patch_size, grid_size=grid_size*sr_scale_factor, in_chans=in_chans)
        self.mask_target = mask_target

        # --------------------------------------------------------------------------
        # Encoder
        embed = PatchEmbed(patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        head = _build_mlp(embed_dim, embed_dim, proj_cfg.mlp_dim, proj_cfg.mlp_depth, out_norm=True)
        self.encoder = VIT(patchify=self.patchify, embed_layer=embed,
                           grid_size=grid_size, embed_dim=embed_dim, num_heads=num_heads, depth=depth,
                           mlp_ratio=mlp_ratio, norm_layer=norm_layer, head=head, mask=False, cls=True,
                           drop=drop, attn_drop=attn_drop, drop_path=drop_path, freeze_pe=freeze_pe)
        # --------------------------------------------------------------------------
        # Target Encoder
        embed = PatchEmbed(patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        head = _build_mlp(embed_dim, embed_dim, proj_cfg.mlp_dim, proj_cfg.mlp_depth, out_norm=True)
        self.target_encoder = VIT(patchify=self.patchify, embed_layer=embed,
                                  grid_size=grid_size, embed_dim=embed_dim, num_heads=num_heads, depth=target_depth,
                                  mlp_ratio=mlp_ratio, norm_layer=norm_layer, head=head, mask=False, cls=True,
                                  drop=drop, attn_drop=attn_drop, drop_path=drop_path, freeze_pe=freeze_pe)

        # --------------------------------------------------------------------------
        # Feature Decoder and Predictor
        embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        head = nn.Linear(decoder_embed_dim, embed_dim, bias=True) # decoder to patch
        self.decoder = CrossDecoder(embed_layer=embed, grid_size=grid_size,
                                    embed_dim=decoder_embed_dim, num_heads=decoder_num_heads, depth=decoder_depth,
                                    mlp_ratio=mlp_ratio, norm_layer=norm_layer, head=head, avg_vis_mask_token=avg_vis_mask_token,
                                    drop=drop, attn_drop=attn_drop, drop_path=drop_path, freeze_pe=freeze_pe)

        # ---------------------------------------------------------------------------
        # HR Gram Teacher
        embed = PatchEmbed(patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)  # Why use a new one?
        head = _build_mlp(embed_dim, embed_dim, proj_cfg.mlp_dim, proj_cfg.mlp_depth, out_norm=True)
        self.hr_gram_teacher = VIT(patchify=self.hr_patchify, embed_layer=embed,
                                   grid_size=grid_size*sr_scale_factor, embed_dim=embed_dim, num_heads=num_heads, depth=target_depth,
                                   mlp_ratio=mlp_ratio, norm_layer=norm_layer, head=head, mask=False, cls=True,
                                   drop=drop, attn_drop=attn_drop, drop_path=drop_path, freeze_pe=freeze_pe)

        self._init_target_encoder_and_hr_gram_teacher()

    @torch.no_grad()
    def _init_target_encoder_and_hr_gram_teacher(self):
        """Initialize Momentum Encoder"""
        for p, q_ema, q2_ema in zip(
            self.encoder.parameters(),
            self.target_encoder.parameters(),
            self.hr_gram_teacher.parameters()
        ):
            q_ema.data[:] = p.data[:]
            q_ema.requires_grad = False

            q2_ema.data[:] = p.data[:]
            q2_ema.requires_grad = False


    @torch.no_grad()
    def _ema_update_trg_enc_and_gram_teacher(self, m):
        """Momentum update of the momentum encoder"""
        for p, q_ema, q2_ema in zip(
            self.encoder.parameters(),
            self.target_encoder.parameters(),
            self.hr_gram_teacher.parameters()
        ):
            q_ema.data = q_ema.data * m + p.data * (1. - m)
            q2_ema.data = q2_ema.data * m + p.data * (1. - m)

    @torch.no_grad()
    def create_views(self, patch_pix):
        N, L, D = patch_pix.shape
        pid = torch.arange(L, device=patch_pix.device).long()

        # sample visible and target patches
        shuffle = torch.argsort(torch.rand(N, L, device=patch_pix.device), dim=1)       # ascend: small is keep, large is remove
        vis_idx = pid[shuffle[:, :self.num_vis]]      # idx of vis patches
        mask_idx = pid[shuffle[:, self.num_vis:]]
        if self.mask_target:
            trg_idx = mask_idx
        else:
            trg_idx = pid[shuffle]

        # gather patches
        vis_pix = patch_pix.gather(dim=1, index=vis_idx[:, :, None].repeat(1, 1, D))
        trg_pix = patch_pix.gather(dim=1, index=trg_idx[:, :, None].repeat(1, 1, D))
        return vis_pix, vis_idx, trg_pix, trg_idx, mask_idx

    @torch.no_grad()
    def scale_masks(self, masks_lr):
        """
        One patch in LR has scale_factor**2 corresponding patches in HR
        This method maps the LR mask to a HR mask so that for each patch idx
        in the LR, it gives the corresponding scale_factor**2 patch indices
        of the HR
        """
        n_patches_lr = self.grid_size
        n_patches_hr = self.grid_size * self.sr_scale_factor

        scaled_masks = []
        for mask_lr in masks_lr:
            hr_indices = []
            for idx_lr in mask_lr:
                # Convert 1D LR index to 2D coordinates
                idx_lr_val = idx_lr.item() if isinstance(idx_lr, torch.Tensor) else int(idx_lr)
                row_lr = idx_lr_val // n_patches_lr
                col_lr = idx_lr_val % n_patches_lr

                # Each LR patch maps to scale_factor x scale_factor HR patches
                for dr in range(self.sr_scale_factor):
                    for dc in range(self.sr_scale_factor):
                        row_hr = row_lr * self.sr_scale_factor + dr
                        col_hr = col_lr * self.sr_scale_factor + dc
                        idx_hr = row_hr * n_patches_hr + col_hr
                        hr_indices.append(idx_hr)

            scaled_masks.append(torch.tensor(hr_indices, device=mask_lr.device, dtype=torch.long))

        return torch.stack(scaled_masks, dim=0)

    def forward_reconstruction_loss(self, pred, target, pred2unit=True):
        if self.loss == 'norm_l2':
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5
            loss = F.mse_loss(pred, target)

        elif self.loss == 'infonce_patches':
            bs, nt, d = pred.shape
            
            if pred2unit:
                pred_mu = pred.mean(1, keepdims=True)
                pred_std = pred.std(1, keepdims=True)
                pred = (pred - pred_mu) / (pred_std + 1e-4)

            pred = F.normalize(pred, p=2, dim=-1)
            target = F.normalize(target, p=2, dim=-1)

            scores = torch.einsum('npd,nqd->npq', pred, target) / self.tau
            labels = torch.arange(nt, dtype=torch.long, device=pred.device)[None].repeat(bs, 1)
            loss = F.cross_entropy(scores.flatten(0, 1), labels.flatten(0, 1)) * (self.tau * 2)

        else:
            raise NotImplementedError(f"Loss type {self.loss} not implemented")
        return loss

    def forward_hr_gram_loss(self, encoder_emb, hr_gram_trg):
        # Downscale hi-res gram targets to lo-res space
        B, Nlr, D = encoder_emb.shape
        Nhr = hr_gram_trg.shape[1]
        if Nhr != Nlr:
            assert Nhr == Nlr * (self.sr_scale_factor**2)
            hr_gram_trg = (hr_gram_trg
                           # (B, Nhr, D) -> (B, Nlr, scale_factor, scale_factor, D)
                           .reshape(B, Nlr, self.sr_scale_factor, self.sr_scale_factor, D)
                           #  (B, Nlr, scale_factor, scale_factor, D) -> (B, Nlr, D, scale_factor, scale_factor)
                           .permute(0, 1, 4, 2, 3)
                           # the interpolate func. expects mini-batch x channels x [optional height] x width
                           .reshape(B * Nlr, D, self.sr_scale_factor, self.sr_scale_factor))
            hr_gram_trg = torch.nn.functional.interpolate(
                hr_gram_trg, size=(1, 1), mode="bilinear", align_corners=False)  # (B*N, D, 1, 1)
            hr_gram_trg = hr_gram_trg.reshape(B, Nlr, D)

        trg = hr_gram_trg.float()
        out = encoder_emb.float()

        if True:  # self.apply_norm:
            trg = F.normalize(trg, dim=-1)
            out = F.normalize(out, dim=-1)

        if self.hr_gram_loss_type == "gram":
            # Compute similarities
            trg_sim = torch.matmul(trg, trg.transpose(-1, -2))
            out_sim = torch.matmul(out, out.transpose(-1, -2))

            return F.mse_loss(out_sim, trg_sim)
        elif self.hr_gram_loss_type == "mse":
            return F.mse_loss(out, trg)
        else:
            raise ValueError('hr_gram_loss_type can only be "gram" or "mse"')

    def forward(self, imgs, hr_img=None, mom=0.99, sim_trg=0.75, update_ema=False):
        # Image to patches
        patch_pix = self.patchify(imgs, self.patch_gap)

        # Create two views by masking
        pix_vis, pid_vis, pix_trg, pid_trg, mask_idx = self.create_views(patch_pix)

        # Get HR vis pix
        if self.use_hr_gram_loss:
            hr_patch_pix = self.hr_patchify(hr_img, self.patch_gap)
            hr_pid_vis = self.scale_masks(pid_vis)
            _, _, D = hr_patch_pix.shape
            hr_pix_vis = hr_patch_pix.gather(dim=1, index=hr_pid_vis[:, :, None].repeat(1, 1, D))

        # Compute targets
        with torch.no_grad():
            # Update the momentum encoder
            if update_ema:
                self._ema_update_trg_enc_and_gram_teacher(mom)

            # Compute reconstruction targets
            Lm = mask_idx.shape[-1]
            x_trg = self.target_encoder(pix_trg, pid_trg)[:, -Lm:]

            # Compute hr_gram targets
            if self.use_hr_gram_loss:
                num_vis_hr_tokens = hr_pid_vis.shape[-1]
                hr_gram_trg = self.hr_gram_teacher(hr_pix_vis, hr_pid_vis)[:, -num_vis_hr_tokens:]

        # Forward encoder
        x_vis = self.encoder(pix_vis, pid_vis)

        # Forward decoder
        x_pred = self.decoder(x_vis, pid_vis, mask_pid=mask_idx)[:, 1:]

        # Compute loss
        recon_loss = self.forward_reconstruction_loss(x_pred, x_trg)

        if self.avg_sim_coeff > 0:
            recon_loss += self.avg_sim_coeff * (avg_pairwise_sim(x_pred, x_pred)-sim_trg).pow(2)
            recon_loss += self.avg_sim_coeff * (avg_pairwise_sim(x_vis[:, 1:], x_vis[:, 1:])-sim_trg).pow(2)

        gram_loss = None
        if self.use_hr_gram_loss:
            gram_loss = self.forward_hr_gram_loss(x_vis[:, 1:], hr_gram_trg)
            loss = recon_loss + gram_loss
        else:
            loss = recon_loss

        # Log metrics
        metrics = {
            'avg_sim_pred': avg_pairwise_sim(x_pred, x_pred).item(),
            'avg_sim_vis': avg_pairwise_sim(x_vis[:, 1:], x_vis[:, 1:]).item(),
            'avg_sim_trg': avg_pairwise_sim(x_trg, x_trg).item(),
        }

        return loss, recon_loss, gram_loss, metrics


def avg_pairwise_sim(q,k):
    q = F.normalize(q, p=2, dim=-1)
    k = F.normalize(k, p=2, dim=-1)

    if len(q.shape) == 3 and len(k.shape) == 3:
        return torch.einsum('npd,nqd->npq', q, k).mean()
    else:
        return torch.einsum('nhpd,nhqd->nhpq', q, k).mean()


CFG = {
    'vit_tiny':   {'embed_dim': 192,  'depth': 12, 'num_heads': 3, 'mlp_ratio': 4, 'patch_size': 16},
    'vit_small':  {'embed_dim': 384,  'depth': 12, 'num_heads': 6, 'mlp_ratio': 4, 'patch_size': 16},
    'vit_base':   {'embed_dim': 768,  'depth': 12, 'num_heads': 12, 'mlp_ratio': 4, 'patch_size': 16},
    'vit_large':  {'embed_dim': 1024, 'depth': 24, 'num_heads': 16, 'mlp_ratio': 4, 'patch_size': 16},
    'vit_huge':   {'embed_dim': 1280, 'depth': 32, 'num_heads': 16, 'mlp_ratio': 4, 'patch_size': 14},
}

def build_lmim(
    backbone, decoder_depth=3, decoder_embed_dim=512, decoder_num_heads=16,
    in_chans=3, use_hr_gram_loss=False, hr_gram_loss_type='gram', sr_scale_factor=2, **kwargs
):
    cfg = CFG[backbone]
    model = LMIM(
        patch_size=cfg['patch_size'], in_chans=in_chans, embed_dim=cfg['embed_dim'], depth=cfg['depth'],
        num_heads=cfg['num_heads'], decoder_embed_dim=decoder_embed_dim, decoder_depth=decoder_depth,
        decoder_num_heads=decoder_num_heads, mlp_ratio=cfg['mlp_ratio'], use_hr_gram_loss=use_hr_gram_loss,
        hr_gram_loss_type=hr_gram_loss_type, sr_scale_factor=sr_scale_factor,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
