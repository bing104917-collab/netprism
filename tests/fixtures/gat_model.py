from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from benchmark.support import predictor_from_forward
from benchmark.support import load_dataset_split, normalised_positions, train_dl_reconstructor


class GATLayer(nn.Module):
  def __init__(self, d_in, d_out, heads=4):
    super().__init__()
    self.heads = heads
    self.d_out = d_out
    self.q = nn.Linear(d_out, d_out * heads)
    self.k = nn.Linear(d_in + 3, d_out * heads)
    self.v = nn.Linear(d_in + 3, d_out * heads)
    self.out = nn.Linear(d_out * heads, d_out)

  def forward(self, q_feat, s_feat, rel_pos):

    B, Nq, K, _ = s_feat.shape
    H = self.heads
    kv_in = torch.cat([s_feat, rel_pos], dim=-1)
    q = self.q(q_feat).view(B, Nq, 1, H, self.d_out)
    k = self.k(kv_in).view(B, Nq, K, H, self.d_out)
    v = self.v(kv_in).view(B, Nq, K, H, self.d_out)
    att = (q * k).sum(-1) / (self.d_out ** 0.5)
    att = att.softmax(dim=2)
    out = (att.unsqueeze(-1) * v).sum(dim=2)
    return self.out(out.reshape(B, Nq, H * self.d_out))


class SensorGAT(nn.Module):
  def __init__(self, d: int = 64, heads: int = 4, k: int = 16):
    super().__init__()
    self.k = k
    self.sensor_enc = nn.Sequential(nn.Linear(3, d), nn.GELU(), nn.Linear(d, d))
    self.query_enc = nn.Sequential(nn.Linear(3, d), nn.GELU(), nn.Linear(d, d))
    self.gat = GATLayer(d, d, heads=heads)
    self.head = nn.Sequential(nn.GELU(), nn.Linear(d, 3))

  def forward(self, q_pos, s_pos, s_val):
    B, Nq, _ = q_pos.shape
    Ns = s_pos.shape[1]

    d_sq = ((q_pos.unsqueeze(2) - s_pos.unsqueeze(1)) ** 2).sum(-1)
    K = min(self.k, Ns)
    _, nbr = d_sq.topk(K, dim=-1, largest=False)
    s_feat = self.sensor_enc(s_val)
    gather = nbr.unsqueeze(-1).expand(-1, -1, -1, s_feat.shape[-1])
    s_feat_k = torch.gather(s_feat.unsqueeze(1).expand(-1, Nq, -1, -1), 2, gather)
    rel = s_pos.unsqueeze(1).expand(-1, Nq, -1, -1)
    rel = torch.gather(rel, 2, nbr.unsqueeze(-1).expand(-1, -1, -1, 3)) - q_pos.unsqueeze(2)
    q_feat = self.query_enc(q_pos)
    out = self.gat(q_feat, s_feat_k, rel)
    return self.head(out + q_feat)


def _forward_factory(coarse_shape):
  Cz, Cy, Cx = coarse_shape

  def forward_fn(model, batch, device):
    tgt = batch["field"].to(device).float()
    fluid = batch["fluid"].to(device).float()
    B, _, Z, Y, X = tgt.shape
    qz = torch.linspace(0, 1, Cz, device=device)
    qy = torch.linspace(0, 1, Cy, device=device)
    qx = torch.linspace(0, 1, Cx, device=device)
    gz, gy, gx = torch.meshgrid(qz, qy, qx, indexing="ij")
    q_pos = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3).unsqueeze(0).expand(B, -1, -1)


    s_pos = normalised_positions(batch["sensor_pos"].to(device).float(), batch)
    s_val = batch["sensor_vals"].to(device).float()
    out = model(q_pos, s_pos, s_val)
    coarse = out.transpose(1, 2).reshape(B, 3, Cz, Cy, Cx)
    pred = F.interpolate(coarse, size=(Z, Y, X), mode="trilinear", align_corners=True)
    return pred, tgt, fluid
  return forward_fn


def _build_model(cfg, device):
  coarse = tuple(cfg.get("coarse_shape", (8, 16, 16)))
  model = SensorGAT(d=int(cfg.get("d", 64)),
           heads=int(cfg.get("heads", 4)),
           k=int(cfg.get("k", 16))).to(device)
  forward_fn = _forward_factory(coarse)
  return model, forward_fn


def load_predictor(cfg, device, checkpoint):
  model, forward_fn = _build_model(cfg, device)
  return predictor_from_forward(model, forward_fn, device, checkpoint)


def train_and_evaluate(cfg: dict, split: str, held_out_city: str,
           seed: int, out_dir: Path, checkpoint: Path | None = None) -> dict:
  torch.manual_seed(seed); np.random.seed(seed)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  ds, splits = load_dataset_split(cfg, split, held_out_city, seed)
  model, forward_fn = _build_model(cfg, device)
  return train_dl_reconstructor(model, ds, splits, cfg, device, out_dir,
                 forward_fn=forward_fn, baseline_name="gat", checkpoint=checkpoint)
