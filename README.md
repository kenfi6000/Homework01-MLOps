# Fashion-MNIST — pipeline MLOps

Application de bout en bout pour entraîner un CNN de type LeNet sur Fashion-MNIST, suivre les expériences dans MLflow, stocker les modèles versionnés dans MinIO et effectuer des prédictions avec FastAPI et Streamlit.

## Architecture et classes

```text
Fashion-MNIST -> entraînement PyTorch -> paramètres, métriques et métadonnées MLflow
                                  -> fashion-minst-bucket/model/<run-id>/model.pt
                                  -> webhook après publication du modèle
Streamlit -> POST /predict -> FastAPI -> modèle téléchargé depuis MinIO
```

Le nom du bucket `fashion-minst-bucket` conserve volontairement l’orthographe demandée (`minst`).
Les dix classes sont celles de Fashion-MNIST : T-shirt/top, Trouser, Pullover, Dress,
Coat, Sandal, Shirt, Sneaker, Bag et Ankle boot.

## Prérequis

- Python 3.11 ou ultérieur
- Docker Desktop avec Docker Compose v2 pour l’environnement complet
- Accès Internet au premier entraînement pour télécharger le jeu Fashion-MNIST

Les identifiants `minioadmin` utilisés comme valeurs de repli dans la configuration sont réservés au développement local. Avant de démarrer les services, configurez des identifiants locaux dans `.env`. Ne réutilisez pas ces valeurs en production.

## Installation locale

Depuis PowerShell à la racine du dépôt :

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

Si PowerShell bloque `Activate.ps1`, aucune activation n’est requise : utilisez directement `.venv\Scripts\python.exe` et `.venv\Scripts\pytest.exe`.

Modifiez `.env` avant de démarrer les services. `.env` est ignoré par Git; `.env.example` ne contient que des valeurs d’exemple.

## Configuration de l’entraînement

Les paramètres par défaut sont définis dans [`configs/training.toml`](configs/training.toml). Chaque valeur peut être remplacée par une variable d’environnement `TRAIN_*` correspondante ou par les options CLI prises en charge.

| Paramètre TOML | Variable d’environnement | CLI | Valeur par défaut |
| --- | --- | --- | --- |
| `epochs` | `TRAIN_EPOCHS` | `--epochs` | 5 |
| `batch_size` | `TRAIN_BATCH_SIZE` | `--batch-size` | 64 |
| `learning_rate` | `TRAIN_LEARNING_RATE` | `--learning-rate` | 0.001 |
| `optimizer` | `TRAIN_OPTIMIZER` | `--optimizer` | `adam` |
| `loss_function` | `TRAIN_LOSS_FUNCTION` | `--loss` | `cross_entropy` |
| `random_seed` | `TRAIN_RANDOM_SEED` | `--seed` | 42 |
| `validation_split` | `TRAIN_VALIDATION_SPLIT` | `--validation-split` | 0.2 |
| `momentum` | `TRAIN_MOMENTUM` | — | 0.9 |
| `weight_decay` | `TRAIN_WEIGHT_DECAY` | — | 0.0001 |
| `learning_rate_schedule` | `TRAIN_LR_SCHEDULE` | — | `none` |
| `early_stopping_patience` | `TRAIN_EARLY_STOPPING_PATIENCE` | — | 5 |
| `data_loader_workers` | `TRAIN_DATA_LOADER_WORKERS` | — | 2 |
| `pin_memory` | `TRAIN_PIN_MEMORY` | — | `true` |

Optimiseurs acceptés : `adam`, `sgd`. Fonctions de perte : `cross_entropy`, `nll_loss`.
Planificateurs : `none`, `step`. La validation rejette les valeurs incompatibles avant
l’entraînement. Le meilleur état du modèle selon la perte de validation est publié.

Exemples :

```powershell
.\.venv\Scripts\python.exe -m fashion_mnist_mlop.train --config configs/training.toml
.\.venv\Scripts\python.exe -m fashion_mnist_mlop.train --epochs 8 --batch-size 128 --learning-rate 0.0005 --optimizer adam --loss cross_entropy --seed 42 --validation-split 0.2
```

Le pipeline enregistre la configuration d’entraînement effective, les pertes train/validation à chaque époque, les précisions de validation, la meilleure époque, et une référence vers le modèle MinIO dans MLflow. Les identifiants et URL sensibles ne sont pas inclus dans les artefacts de configuration MLflow.

## Démarrage complet avec Docker Compose

À la racine du dépôt :

```powershell
Copy-Item .env.example .env
```

Dans `.env`, remplacez `MINIO_ACCESS_KEY` et `MINIO_SECRET_KEY` par des identifiants locaux personnels. Construisez les images une fois :

```powershell
docker compose build
```

Puis démarrez MinIO, MLflow et le récepteur webhook :

```powershell
docker compose up -d minio mlflow webhook-receiver
docker compose ps
```

Lancez l’entraînement comme service Compose ponctuel. Compose attend que MinIO, MLflow et le récepteur webhook soient prêts. Le conteneur terminé reste présent afin que Compose puisse vérifier sa réussite avant de démarrer l’API :

```powershell
docker compose up train-model
```

À la fin, le terminal affiche l’identifiant MLflow, la précision de validation et la clé objet publiée. Démarrez ensuite l’API et Streamlit sans reconstruire ni relancer l’entraînement :

```powershell
docker compose up -d api streamlit
docker compose ps
```

Les images d’application installent les distributions CPU de PyTorch et TorchVision afin d’éviter d’inclure les bibliothèques CUDA dans les conteneurs.

Adresses locales :

| Service | Adresse |
| --- | --- |
| Streamlit | <http://127.0.0.1:8501> |
| API FastAPI | <http://127.0.0.1:8000> |
| Documentation OpenAPI | <http://127.0.0.1:8000/docs> |
| MLflow | <http://127.0.0.1:5000> |
| Console MinIO | <http://127.0.0.1:9001> |
| Récepteur webhook | <http://127.0.0.1:8001/health> |

Les ports publiés par Compose sont limités à la machine locale. Le service MinIO utilise l’image Bitnami archivée disponible pour l’environnement de développement; elle n’est plus maintenue. Pour une utilisation durable ou en production, choisissez une distribution MinIO prise en charge et configurez les identifiants, TLS, stockage et accès réseau selon votre environnement.

Les données MinIO et MLflow sont persistées dans les volumes Compose `minio_storage` et `mlflow_data`. Le service d’entraînement doit réussir avant le démarrage de l’API; Streamlit appelle l’API sur le réseau Compose. Pour afficher les journaux : `docker compose logs -f train-model api streamlit`.

## MLflow

L’interface MLflow est disponible à <http://127.0.0.1:5000>. Ouvrez l’expérience `fashion-mnist-classifier` pour consulter les paramètres, les pertes par époque, la précision de validation, `training_config.json` et les métadonnées pointant vers le modèle MinIO.

Le backend MLflow et ses artefacts sont stockés dans le volume persistant `mlflow_data`.

## MinIO et sélection du modèle

Les variables principales sont `MINIO_ENDPOINT`, `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY`,
`MINIO_SECURE`, `MODEL_BUCKET_NAME` et `MODEL_PREFIX`. Par défaut, le bucket est
`fashion-minst-bucket` et le préfixe `model/`. Le bucket est créé au téléversement s’il
n’existe pas. Chaque entraînement publie le meilleur état validé sous
`model/<run-id>/model.pt`, sans remplacer les versions précédentes.

Au démarrage, l’API charge `MODEL_OBJECT_KEY` si cette variable est définie; sinon, elle
sélectionne l’objet MinIO le plus récent sous `MODEL_PREFIX`. `MODEL_PATH` peut être défini
pour un test local uniquement et prend priorité sur MinIO.

## Webhook de fin d’entraînement

Le webhook n’est envoyé qu’après la fin de l’entraînement, la journalisation MLflow et le
téléversement réussi du modèle. Son URL et son activation sont réglables avec
`WEBHOOK_URL` et `WEBHOOK_ENABLED`. Le récepteur Compose écoute sur
`http://127.0.0.1:8001/webhook`; consultez ses journaux avec
`docker compose logs webhook-receiver`.

Exemple de payload :

```json
{
  "status": "success",
  "run_id": "<mlflow-run-id>",
  "bucket": "fashion-minst-bucket",
  "object_key": "model/<run-id>/model.pt",
  "validation_accuracy": 0.87,
  "experiment_name": "fashion-mnist-classifier"
}
```

## API et interface

- `GET /health` indique si l’API a pu charger un modèle.
- `POST /predict` accepte un unique fichier image multipart dans le champ `file`; la réponse inclut la classe, la confiance et les probabilités par classe.

Exemple PowerShell :

```powershell
curl.exe -F "file=@tests\fashion_mnist_samples\6_shirt_01.png;type=image/png" http://127.0.0.1:8000/predict
```

L’API valide les fichiers vides, le type MIME et le contenu image. Si MinIO ou le modèle
est indisponible, `/health` renvoie un état dégradé et `/predict` explique que le modèle
n’est pas chargé. Streamlit affiche l’image, transmet ses octets à FastAPI et présente
la classe prédite avec son niveau de confiance; il ne fait pas d’inférence localement.

Les exemples 28 × 28 en niveaux de gris dans [`tests/fashion_mnist_samples/`](tests/fashion_mnist_samples/)
proviennent du jeu de test officiel. Ils sont documentés dans le README de ce dossier.
Le classifieur est destiné aux images au style Fashion-MNIST, et non aux photographies
de vêtements en conditions réelles.

## Tests et validation locale

```powershell
.\.venv\Scripts\pytest.exe tests -q
```

Les tests couvrent le chargement et la validation de configuration, le prétraitement
28 × 28, les entrées invalides de l’API et le chargement des poids locaux pour les
tests automatisés.

## Nettoyage des services

Arrêter les conteneurs sans supprimer les données persistées :

```powershell
docker compose down
```

Ne supprimez les volumes persistants (`docker compose down -v`) que si vous souhaitez
effacer les modèles MinIO, les expériences MLflow et les autres données associées.
