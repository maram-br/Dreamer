"""
pretrain_world_model_offline.py
================================================================================
PrÃ©-entraÃ®ne UNIQUEMENT la partie "dynamique pure" du WorldModel
(`encoder`, `gru`, `next_state_head`, `recon_head`, et le `obs_norm` qui les
prÃ©cÃ¨de) sur des transitions de conduite "normale" collectÃ©es hors-ligne
(collect_carla_driving_data.py, rÃ©el ou --mock).

Usage:
  python pretrain_world_model_offline.py --data carla_normal_driving.npz \
      --out wm_pretrained.pt --epochs 30

Le fichier produit (wm_pretrained.pt) est un state_dict de WorldModel
directement utilisable avec :
  python run_training.py --mode mock --wm_checkpoint wm_pretrained.pt
(qui passe maintenant rÃ©ellement par build_agent(wm_state=...), cf. le fix
prÃ©cÃ©dent de run_training.py / sdbs_dreamer.py)
================================================================================
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def load_dataset(path: str) -> dict:
    npz = np.load(path)
    data = {k: npz[k] for k in ("states", "actions", "next_states")}
    n = data["states"].shape[0]
    print(f"[pretrain] {n} transitions chargÃ©es depuis {path} "
          f"(state_dim={data['states'].shape[1]}, "
          f"action_dim={data['actions'].shape[1]})")
    return data


def train_val_split(data: dict, val_frac: float = 0.1, seed: int = 0):
    n = data["states"].shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_val = max(1, int(n * val_frac))
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    train = {k: v[train_idx] for k, v in data.items()}
    val = {k: v[val_idx] for k, v in data.items()}
    return train, val


def pretrain(data_path: str, out_path: str, epochs: int = 30,
            batch_size: int = 256, lr: float = 1e-3, device: str = "cpu",
            hidden: int = 256, latent: int = 128, seed: int = 0) -> str:
    if not HAS_TORCH:
        raise RuntimeError("PyTorch requis pour le prÃ©-entraÃ®nement.")

    from sdbs.model.sdbs_dreamer import WorldModel

    torch.manual_seed(seed)
    np.random.seed(seed)

    data = load_dataset(data_path)
    state_dim = data["states"].shape[1]
    action_dim = data["actions"].shape[1]
    train_data, val_data = train_val_split(data, val_frac=0.1, seed=seed)

    wm = WorldModel(state_dim, action_dim, hidden=hidden, latent=latent).to(device)

    # R8 Â· on n'entraÃ®ne QUE la dynamique pure : encoder + obs_norm + gru +
    # next_state_head + recon_head. Les autres tÃªtes (reward/risk/progress/
    # density) restent Ã  leur init alÃ©atoire -- on ne calcule volontairement
    # AUCUNE loss sur elles ici, donc elles ne reÃ§oivent aucun gradient (pas
    # besoin de geler leurs paramÃ¨tres : sans loss, pas de backward sur ces
    # branches). On ne passe Ã  l'optimizer QUE les paramÃ¨tres concernÃ©s, pour
    # que ce soit explicite et que l'intention ne dÃ©pende pas d'un dÃ©tail
    # d'implÃ©mentation du graphe d'autograd.
    trainable_modules = [wm.obs_norm, wm.encoder, wm.gru, wm.next_state_head,
                         wm.recon_head]
    trainable_params = [p for m in trainable_modules for p in m.parameters()]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)

    def _to_t(x):
        return torch.as_tensor(x, dtype=torch.float32, device=device)

    n_train = train_data["states"].shape[0]
    history = []
    for epoch in range(epochs):
        perm = np.random.permutation(n_train)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n_train, batch_size):
            mb_idx = perm[start:start + batch_size]
            s = _to_t(train_data["states"][mb_idx])
            a = _to_t(train_data["actions"][mb_idx])
            ns = _to_t(train_data["next_states"][mb_idx])

            # R5 Â· alimenter le RunningNorm avec les vrais Ã©tats observÃ©s,
            # comme pendant collect_rollout() dans sdbs_dreamer.py, pour que
            # les statistiques transfÃ©rÃ©es au pipeline en ligne soient dÃ©jÃ 
            # correctement calibrÃ©es sur de la conduite rÃ©elle/normale.
            with torch.no_grad():
                wm.obs_norm.update(s)

            out = wm(s, a)
            loss_state = nn.functional.mse_loss(out["next_state"], ns)
            loss_recon = nn.functional.mse_loss(out["recon"], s)
            loss = loss_state + 0.5 * loss_recon

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            epoch_loss += float(loss.item())
            n_batches += 1

        train_loss = epoch_loss / max(1, n_batches)

        # ---- validation (pas de mise Ã  jour de obs_norm ici) ----
        wm.eval()
        with torch.no_grad():
            vs = _to_t(val_data["states"])
            va = _to_t(val_data["actions"])
            vns = _to_t(val_data["next_states"])
            vout = wm(vs, va)
            val_loss_state = nn.functional.mse_loss(vout["next_state"], vns).item()
            val_mae_state = (vout["next_state"] - vns).abs().mean().item()
        wm.train()

        history.append(dict(epoch=epoch, train_loss=train_loss,
                            val_loss_state=val_loss_state,
                            val_mae_state=val_mae_state))
        print(f"epoch {epoch:03d} | train_loss={train_loss:.4f} | "
              f"val_loss_state={val_loss_state:.4f} | "
              f"val_mae_next_state={val_mae_state:.4f}")

    torch.save(wm.state_dict(), out_path)
    print(f"\n[pretrain] WorldModel prÃ©-entraÃ®nÃ© sauvegardÃ© -> {out_path}")
    print("Pour l'utiliser dans le pipeline RL :")
    print(f"  python run_training.py --mode mock --wm_checkpoint {out_path}")
    return out_path


def main():
    p = argparse.ArgumentParser(description="PrÃ©-entraÃ®nement offline du "
                                            "WorldModel (dynamique pure)")
    p.add_argument("--data", required=True,
                   help="Fichier .npz produit par collect_carla_driving_data.py")
    p.add_argument("--out", default="wm_pretrained.pt")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="cpu")
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--latent", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    pretrain(args.data, args.out, epochs=args.epochs,
             batch_size=args.batch_size, lr=args.lr, device=args.device,
             hidden=args.hidden, latent=args.latent, seed=args.seed)


if __name__ == "__main__":
    main()
