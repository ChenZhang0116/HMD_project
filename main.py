from HatefulMemesModel import HatefulMemesModel
from pathlib import Path

data_dir = Path.cwd().parent / "data"
img_path = data_dir / "img"
train_path = data_dir / "train.jsonl"
dev_unseen_path = data_dir / "dev_unseen.jsonl"
test_unseen_path = data_dir / "test_unseen.jsonl"
dev_seen_path = data_dir / "dev_seen.jsonl"
test_seen_path = data_dir / "test_seen.jsonl"

output_path = Path.cwd() / "model-outputs"

hparams = {

    # Required hparams
    "train_path": train_path,
    "dev_path": dev_unseen_path,
    "test_path": test_unseen_path,
    "img_dir": data_dir,

    # Optional hparams
    "embedding_dim": 150,
    "language_feature_dim": 300,
    "vision_feature_dim": 300,
    "fusion_output_size": 256,
    "output_path": output_path,
    "dev_limit": None,
    "lr": 0.00005,
    "max_epochs": 10,
    "n_gpu": 1,
    "batch_size": 4,
    # allows us to "simulate" having larger batches
    "accumulate_grad_batches": 16,
    "early_stop_patience": 3,
}

hateful_memes_model = HatefulMemesModel(hparams=hparams)
hateful_memes_model.fit()

