"""
traffic_predictor.py
================================================================================
Prédiction multi-agent pour VRU et véhicules.
================================================================================
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# 0.  Helpers collision-risk
# ==============================================================================

def compute_collision_risk(ego_traj, agent_predictions, sigma: float = 1.5) -> float:
    """Risque de collision [0,1] entre la trajectoire ego et les agents prédits.

    Identique à la version de référence — conservé pour la compatibilité du
    SDBSPlanner existant.
    """
    ego = np.asarray(ego_traj, dtype=np.float32).reshape(-1, 2)
    risk = 0.0
    for pred in agent_predictions.values():
        traj = pred[0] if isinstance(pred, (tuple, list)) else pred
        traj = np.asarray(traj, dtype=np.float32).reshape(-1, 2)
        t = min(len(ego), len(traj))
        for i in range(t):
            d = float(np.linalg.norm(ego[i] - traj[i]))
            risk = max(risk, float(np.exp(-(d * d) / (2.0 * sigma * sigma))))
    return float(np.clip(risk, 0.0, 1.0))


def compute_collision_risk_weighted(
    ego_traj,
    agent_predictions,
    sigma: float = 1.5,
    crossing_idx: int = 0,     # indice de l'intention "traverse" dans intention_probs
) -> float:
    """Variante qui pondère chaque agent par sa probabilité de traverser.

    Évite de pénaliser les plans qui croisent des piétons qui longent le
    trottoir sans intention de traverser — contribution absente du papier.

    agent_predictions : dict agent_id -> (traj, uncertainty, intention_probs)
      ou le format de référence (traj,) / (traj, uncertainty).
    """
    ego = np.asarray(ego_traj, dtype=np.float32).reshape(-1, 2)
    risk = 0.0
    for pred in agent_predictions.values():
        if isinstance(pred, (tuple, list)) and len(pred) >= 3:
            traj, _, intent = pred[0], pred[1], pred[2]
            # intent = np.ndarray shape (n_intents,) ou None
            if intent is not None:
                crossing_w = float(np.asarray(intent).reshape(-1)[crossing_idx])
            else:
                crossing_w = 1.0
        else:
            traj = pred[0] if isinstance(pred, (tuple, list)) else pred
            crossing_w = 1.0
        traj = np.asarray(traj, dtype=np.float32).reshape(-1, 2)
        t = min(len(ego), len(traj))
        for i in range(t):
            d = float(np.linalg.norm(ego[i] - traj[i]))
            raw = float(np.exp(-(d * d) / (2.0 * sigma * sigma)))
            risk = max(risk, crossing_w * raw)
    return float(np.clip(risk, 0.0, 1.0))


# ==============================================================================
# 1.  TrafficPredictor (agent unique)
# ==============================================================================

# Intentions sémantiques modélisées
INTENT_CROSS   = 0   # traverse la chaussée
INTENT_WALK    = 1   # longe le trottoir
INTENT_STOP    = 2   # s'arrête / attend
N_INTENTS      = 3


class TrafficPredictor(nn.Module):
    """LSTM + MLP qui prédit horizon positions / vitesses + intention pour un agent.

    Améliorations vs référence :
      * IntentionHead  : classe sémantique du comportement probable.
      * Dropout        : activé en inférence pour l'incertitude épistémique.
      * Prédiction résiduelle bornée (delta depuis la dernière position observée).
    """

    def __init__(self, state_dim: int, agent_state_dim: int = 5,
                 horizon: int = 8, hidden_dim: int = 128,
                 max_speed: float = 15.0, pos_bound: float = 100.0,
                 dropout: float = 0.1, device: str = "cpu"):
        super().__init__()
        self.state_dim      = state_dim
        self.agent_state_dim = agent_state_dim
        self.horizon        = horizon
        self.hidden_dim     = hidden_dim
        self.max_speed      = max_speed
        self.pos_bound      = pos_bound
        self.device         = torch.device(device)

        self.encoder   = nn.LSTM(agent_state_dim, hidden_dim, batch_first=True)
        self.dropout   = nn.Dropout(p=dropout)
        self.head_traj = nn.Linear(hidden_dim, horizon * 2)
        self.head_vel  = nn.Linear(hidden_dim, horizon * 2)
        self.head_unc  = nn.Linear(hidden_dim, horizon * 2)
        # Nouvelle tête : intention sémantique
        self.head_intent = nn.Linear(hidden_dim, N_INTENTS)

        self.to(self.device)

    def forward(self, agent_history):
        """agent_history : (B, seq_len, agent_state_dim).

        Retourne (pred_traj, pred_vel, uncertainty, intent_probs)
          chacun  (B, horizon, 2) sauf intent_probs (B, N_INTENTS).
        """
        if not torch.is_tensor(agent_history):
            agent_history = torch.as_tensor(
                np.asarray(agent_history, dtype=np.float32))
        agent_history = agent_history.float().to(self.device)
        if agent_history.dim() == 2:
            agent_history = agent_history.unsqueeze(0)

        _, (h_n, _) = self.encoder(agent_history)
        h = self.dropout(h_n[-1])          # (B, hidden_dim)
        b = h.shape[0]

        last_pos = agent_history[:, -1, 0:2]
        delta    = (torch.tanh(self.head_traj(h)) * self.pos_bound).view(b, self.horizon, 2)
        pred_traj = last_pos.unsqueeze(1) + delta

        pred_vel    = (torch.tanh(self.head_vel(h)) * self.max_speed).view(b, self.horizon, 2)
        uncertainty = torch.sigmoid(self.head_unc(h)).view(b, self.horizon, 2)
        intent_probs = F.softmax(self.head_intent(h), dim=-1)    # (B, N_INTENTS)

        return pred_traj, pred_vel, uncertainty, intent_probs

    def loss(self, pred_traj, pred_vel, target_traj, target_vel,
             intent_logits=None, intent_labels=None) -> torch.Tensor:
        """Perte combinée trajectoire + (optionnellement) intention."""
        l = F.mse_loss(pred_traj, target_traj) + F.mse_loss(pred_vel, target_vel)
        if intent_logits is not None and intent_labels is not None:
            l = l + 0.3 * F.cross_entropy(intent_logits, intent_labels)
        return l

    @torch.no_grad()
    def predict_single(self, history, n_steps: int = None):
        """Prédit les positions futures d'un agent. Retourne (n_steps, 2)."""
        n_steps = n_steps or self.horizon
        pred_traj, _, _, _ = self.forward(history)
        n = min(n_steps, self.horizon)
        return pred_traj[0, :n].cpu().numpy()

    def mc_uncertainty(self, agent_history, n_passes: int = 5) -> np.ndarray:
        """Incertitude épistémique MC-dropout — active le dropout en inférence.

        Retourne la variance sur les prédictions de position (horizon, 2).
        """
        self.train()    # active dropout
        samples = []
        with torch.no_grad():
            for _ in range(n_passes):
                traj, _, _, _ = self.forward(agent_history)
                samples.append(traj[0].cpu().numpy())   # (horizon, 2)
        self.eval()
        return np.var(np.stack(samples, axis=0), axis=0)   # (horizon, 2)


# ==============================================================================
# 2.  MultiAgentPredictor
# ==============================================================================

class MultiAgentPredictor:
    """Suit plusieurs agents et prédit tous leurs futurs en une passe batch.

    Améliorations vs référence :
      * predict_and_score : retourne (traj, uncertainty_combinée, intent_probs)
        → directement consommable par le SDBSPlanner et compute_collision_risk_weighted.
      * Incertitude combinée = aléatoire (σ) + épistémique (MC-dropout variance).
    """

    def __init__(self, state_dim: int, max_agents: int = 10,
                 horizon: int = 8, hidden_dim: int = 128,
                 seq_len: int = 5, mc_passes: int = 5,
                 device: str = "cpu"):
        self.state_dim  = state_dim
        self.max_agents = max_agents
        self.seq_len    = seq_len
        self.mc_passes  = mc_passes
        self.device     = torch.device(device)
        self.predictor  = TrafficPredictor(
            state_dim, horizon=horizon, hidden_dim=hidden_dim, device=device)
        self.histories : dict[int, list] = {}
        self.agent_types: dict[int, int] = {}

    # ---- suivi d'agents ----
    def add_agent(self, agent_id: int, agent_type: int = 0) -> None:
        if agent_id not in self.histories:
            self.histories[agent_id] = []
        self.agent_types[agent_id] = int(agent_type)

    def observe_agent(self, agent_id: int, position, velocity) -> None:
        if agent_id not in self.histories:
            self.add_agent(agent_id)
        cls = self.agent_types.get(agent_id, 0)
        self.histories[agent_id].append([
            float(position[0]), float(position[1]),
            float(velocity[0]), float(velocity[1]), float(cls),
        ])
        if len(self.histories[agent_id]) > self.seq_len:
            self.histories[agent_id] = self.histories[agent_id][-self.seq_len:]

    def _padded(self, hist) -> np.ndarray:
        hist = list(hist)
        if not hist:
            return np.zeros((self.seq_len, 5), dtype=np.float32)
        while len(hist) < self.seq_len:
            hist.insert(0, hist[0])
        return np.asarray(hist[-self.seq_len:], dtype=np.float32)

    # ---- prédiction de base (compatible référence) ----
    @torch.no_grad()
    def predict_all(self) -> dict:
        """Retourne dict agent_id -> (trajectory, velocity, uncertainty)."""
        ids = [a for a, h in self.histories.items() if h]
        if not ids:
            return {}
        batch = np.stack([self._padded(self.histories[a]) for a in ids])
        traj, vel, unc, _ = self.predictor(batch)
        traj, vel, unc = traj.cpu().numpy(), vel.cpu().numpy(), unc.cpu().numpy()
        return {a: (traj[i], vel[i], unc[i]) for i, a in enumerate(ids)}

    # ---- prédiction enrichie (nouvelle contribution) ----
    def predict_and_score(self) -> dict:
        """Retourne dict agent_id -> (traj, combined_uncertainty, intent_probs).

        combined_uncertainty : (horizon, 2) — combine aléatoire + MC-dropout.
        intent_probs         : (N_INTENTS,) — probabilité de chaque intention.

        Ce format est consommable par :
          * compute_collision_risk_weighted  (pondération par P(INTENT_CROSS))
          * SDBSPlanner._score_plan          (via collision_risk_weighted)
          * BudgetConfig difficulty estimate (incertitude épistémique)
        """
        ids = [a for a, h in self.histories.items() if h]
        if not ids:
            return {}
        batch_np = np.stack([self._padded(self.histories[a]) for a in ids])

        # Passe principale (déterministe)
        with torch.no_grad():
            traj, vel, unc_aleatoire, intent = self.predictor(batch_np)
        traj_np    = traj.cpu().numpy()
        unc_al_np  = unc_aleatoire.cpu().numpy()
        intent_np  = intent.cpu().numpy()   # (n_agents, N_INTENTS)

        # Incertitude épistémique MC-dropout
        out = {}
        for i, a_id in enumerate(ids):
            hist_i = self._padded(self.histories[a_id])
            unc_ep = self.predictor.mc_uncertainty(hist_i, n_passes=self.mc_passes)
            # Combine : max-fusion (prend le pire cas)
            unc_combined = np.maximum(unc_al_np[i], unc_ep)
            out[a_id] = (traj_np[i], unc_combined, intent_np[i])
        return out

    # ---- risque de collision (compatible référence) ----
    def get_collision_risk(self, ego_pos, ego_traj, agent_predictions) -> float:
        traj = ego_traj if ego_traj is not None else [ego_pos]
        return compute_collision_risk(traj, agent_predictions)

    # ---- risque pondéré par intention (nouveau) ----
    def get_collision_risk_weighted(self, ego_traj, agent_predictions) -> float:
        """Risque pondéré par la probabilité d'intention de traversée."""
        return compute_collision_risk_weighted(ego_traj, agent_predictions)