# S-DBS Dreamer-PPO

Planner *Serendipitous & Diverse Beam Search* (S-DBS) au-dessus d'un Dreamer-PPO,
pour la conduite autonome consciente des VRU (piétons/cyclistes), avec un mode
mock (NumPy pur, sans CARLA) pour le développement local et un mode CARLA réel
pour l'entraînement final.


## Sur le serveur CARLA

⚠️ Cette section ne s'exécute QUE sur la machine où tourne le serveur CARLA
(le module `carla` n'est pas installable sur un PC sans CARLA — voir
`requirements.txt`).

### 1. Vérifier que tout est en place

```bash
# Le serveur CARLA doit déjà tourner (CarlaUE4.exe / CarlaUE4.sh) avant ces commandes
python3 -c "import carla; print('OK, module carla trouvé')"
python3 -c "
import carla
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
print('Version serveur :', client.get_server_version())
print('Carte chargée   :', client.get_world().get_map().name)
"
```

Si la première commande échoue avec `ModuleNotFoundError`, installe le module
depuis l'installation CARLA elle-même (pas de paquet PyPI générique) :

```bash
pip install <CARLA_ROOT>/PythonAPI/carla/dist/carla-<version>-<tag>.whl
```

### 2. Copier le projet sur le serveur

```bash
# Depuis ton PC (ou git clone / scp selon ton setup)
scp -r sdbsv2/ user@serveur:/chemin/vers/sdbsv2/
ssh user@serveur
cd /chemin/vers/sdbsv2/
pip install -r requirements.txt   # numpy / torch / pygame (pygame optionnel sur le serveur)
pip install -e .                  # OBLIGATOIRE : rend le package `sdbs` importable
                                  # (sinon les scripts échouent avec
                                  #  ModuleNotFoundError: No module named 'sdbs')
```

> À défaut de `pip install -e .`, préfixe chaque commande par `PYTHONPATH=.`
> (les scripts n'ajoutent que `scripts/` au path, pas la racine du dépôt).

### 3. Collecter de la conduite "normale" via le Traffic Manager (autopilote)

Pas besoin d'agent entraîné — le Traffic Manager pilote l'ego pendant que le
script enregistre les transitions dans le layout 20-dim compatible avec
`EnhancedMockEnv`.

```bash
python scripts/collect_carla_driving_data.py \
    --host localhost --port 2000 \
    --town Town03 \
    --episodes 30 --steps_per_episode 600 \
    --n_vehicles 15 --n_walkers 20 \
    --out data/carla_normal.npz
```

### 4. Pré-entraîner le World Model (dynamique pure) sur ces données réelles

```bash
python scripts/pretrain_world_model_offline.py \
    --data data/carla_normal.npz \
    --out checkpoints/wm_pretrained_carla.pt \
    --epochs 30 --device cuda
```


### 5. Lancer le training RL avec ce World Model pré-entraîné

```bash
python scripts/run_training.py \
    --mode carla --device cuda \
    --wm_checkpoint checkpoints/wm_pretrained_carla.pt \
    --iters 1000 \
    --eval_every 25 \
    --save_every 50 --save_path checkpoints/sdbs_checkpoint.pt \
    --traffic_predictor
```


### 6. Reprendre un training interrompu

```bash
python scripts/run_training.py \
    --mode carla --device cuda \
    --resume checkpoints/sdbs_checkpoint.pt \
    --iters 1000 \
    --save_every 50 --save_path checkpoints/sdbs_checkpoint.pt
```

`--resume` recharge aussi les optimizers + l'état du curriculum + les RNG
(contrairement à `--wm_checkpoint`, qui ne charge que des poids de réseau).

### 7. Visualiser les résultats (rapatrier le checkpoint sur ton PC)

```bash
# Sur le serveur
scp checkpoints/sdbs_checkpoint.pt user@ton-pc:/chemin/local/

# Sur ton PC (pas besoin de carla pour ça)
python sdbs/viz/visualize_pg.py --checkpoint checkpoints/sdbs_checkpoint.pt
```

---

## Installation locale (PC, sans CARLA)

```bash
pip install -r requirements.txt
```


## Développement local (mode mock)

```bash
# Smoke test rapide (3 itérations)
python run_training.py --smoke

# Training mock complet
python run_training.py --mode mock --iters 100 --device cpu

# Comparaison Greedy vs S-DBS (sans rien entraîné, juste les stubs NumPy)
python run_training.py --ablation

# Visualisation pygame du greedy trap
python visualize_greedy_vs_sdbs.py
python visualize_greedy_vs_sdbs.py --checkpoint checkpoint.pt   # avec ton modèle

# Pré-entraînement offline du World Model (dynamique pure), sans CARLA
python collect_carla_driving_data.py --mock --episodes 20 --steps_per_episode 200 \
    --out data/mock_normal.npz
python pretrain_world_model_offline.py --data data/mock_normal.npz \
    --out wm_pretrained.pt --epochs 30
python run_training.py --mode mock --wm_checkpoint wm_pretrained.pt
```

---

## Structure du projet

| Fichier | Rôle |
|---|---|
| `sdbs_core.py` | Planner S-DBS, PER, curriculum, GAE — NumPy pur, testable sans torch |
| `sdbs_dreamer.py` | Réseaux PyTorch (ActorCritic, WorldModel+GRU), boucle d'entraînement |
| `enhanced_mock_env.py` | Environnement mock 20-dim (greedy trap, occlusion, scénarios par tier) |
| `run_training.py` | Point d'entrée CLI (mock / carla / smoke / ablation) |
| `visualize_greedy_vs_sdbs.py` | Replay pygame Greedy vs S-DBS |
| `carla_obs_utils.py` | Extraction d'observation 20-dim depuis un monde CARLA réel |
| `collect_carla_driving_data.py` | Collecte de conduite via Traffic Manager (réel ou `--mock`) |
| `pretrain_world_model_offline.py` | Pré-entraînement offline de la dynamique du World Model |
| `requirements.txt` | Dépendances (note : `carla` exclu, serveur uniquement) |