import json
import logging
import random
import tempfile
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm

import fasttext
import torch
import torchvision
from matplotlib import Path

import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning import callbacks
from pytorch_lightning.core import optimizer

from HatefulMemesDataset import HatefulMemesDataset
from LanguageAndVisionConcat import LanguageAndVisionConcat

warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.WARNING)

class HatefulMemesModel(LightningModule):
    def __init__(self, hparams):
        for data_key in ["train_path", "dev_path", "img_dir", ]:
            # ok, there's one for-loop but it doesn't count
            if data_key not in hparams.keys():
                raise KeyError(
                    f"{data_key} is a required hparam in this model"
                )

        super(HatefulMemesModel, self).__init__()
        self.save_hyperparameters()
        #self.hparams = hparams
        self.hparams.update(hparams)

        # assign some hparams that get used in multiple places
        self.embedding_dim = self.hparams.get("embedding_dim", 300)
        self.language_feature_dim = self.hparams.get(
            "language_feature_dim", 300
        )
        self.vision_feature_dim = self.hparams.get(
            # balance language and vision features by default
            "vision_feature_dim", self.language_feature_dim
        )
        self.output_path = Path(
            self.hparams.get("output_path", "model-outputs")
        )
        self.output_path.mkdir(exist_ok=True)

        # instantiate transforms, datasets
        self.text_transform = self._build_text_transform()
        self.image_transform = self._build_image_transform()
        self.train_dataset = self._build_dataset("train_path")
        self.dev_dataset = self._build_dataset("dev_path")

        # set up model and training
        self.model = self._build_model()
        self.trainer_params = self._get_trainer_params()

    ## Required LightningModule Methods (when validating) ##

    def forward(self, text, image, label=None):
        return self.model(text, image, label)

    def training_step(self, batch, batch_nb):
        preds, loss = self.forward(
            text=batch["text"],
            image=batch["image"],
            label=batch["label"]
        )

        return {"loss": loss}

    def validation_step(self, batch, batch_nb):
        preds, loss = self.eval().forward(
            text=batch["text"],
            image=batch["image"],
            label=batch["label"]
        )
        self.log('val_loss', loss)
        return {"batch_val_loss": loss}

    def validation_epoch_end(self, outputs):
        val_loss = torch.stack(tuple(output["batch_val_loss"] for output in outputs)).mean()

        return {
            'val_loss': val_loss,
            'progress_bar': {'val_loss': val_loss}
        }

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            params=self.parameters(),
            lr=self.hparams.get("lr", 0.001)
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=1,
            verbose=True
        )
        return {
            'optimizer': optimizer,
            'scheduler': scheduler,
            'monitor': 'val_loss'
        }

    # def configure_optimizers(self):
    #     optimizers = [
    #         torch.optim.AdamW(
    #             self.model.parameters(),
    #             lr=self.hparams.get("lr", 0.001)
    #         )
    #     ]
    #     schedulers = [
    #         torch.optim.lr_scheduler.ReduceLROnPlateau(
    #             optimizers[0],
    #             verbose=True
    #         )
    #     ]
    #     return optimizers, schedulers

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.hparams.get("batch_size", 4),
            num_workers=self.hparams.get("num_workers", 16)
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dev_dataset,
            shuffle=False,
            batch_size=self.hparams.get("batch_size", 4),
            num_workers=self.hparams.get("num_workers", 16)
        )

    ## Convenience Methods ##

    def fit(self):
        self._set_seed(self.hparams.get("random_state", 42))
        self.trainer = Trainer(**self.trainer_params)
        self.trainer.fit(self)

    def _set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _build_text_transform(self):
        with tempfile.NamedTemporaryFile() as ft_training_data:
            ft_path = Path(ft_training_data.name)
            with ft_path.open("w") as ft:
                training_data = [
                    json.loads(line)["text"] + "/n"
                    for line in open(
                        self.hparams.get("train_path")
                    ).read().splitlines()
                ]
                for line in training_data:
                    ft.write(line + "\n")
                language_transform = fasttext.train_unsupervised(
                    str(ft_path),
                    model=self.hparams.get("fasttext_model", "cbow"),
                    dim=self.embedding_dim
                )
        return language_transform

    def _build_image_transform(self):
        image_dim = self.hparams.get("image_dim", 224)
        image_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(
                    size=(image_dim, image_dim)
                ),
                torchvision.transforms.ToTensor(),
                # all torchvision models expect the same
                # normalization mean and std
                # https://pytorch.org/docs/stable/torchvision/models.html
                torchvision.transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)
                ),
            ]
        )
        return image_transform

    def _build_dataset(self, dataset_key):

        if dataset_key == "train_path" or dataset_key == "dev_path":
            data_dir = self.hparams.get(dataset_key)
        else:
            data_dir = dataset_key

        return HatefulMemesDataset(
            data_path=data_dir,
            img_dir=self.hparams.get("img_dir"),
            image_transform=self.image_transform,
            text_transform=self.text_transform,
            # limit training samples only
            dev_limit=(
                self.hparams.get("dev_limit", None)
                if "train" in str(dataset_key) else None
            ),
            balance=True if "train" in str(dataset_key) else False,
        )

    def _build_model(self):
        # we're going to pass the outputs of our text
        # transform through an additional trainable layer
        # rather than fine-tuning the transform
        language_module = torch.nn.Linear(
            in_features=self.embedding_dim,
            out_features=self.language_feature_dim
        )

        # easiest way to get features rather than
        # classification is to overwrite last layer
        # with an identity transformation, we'll reduce
        # dimension using a Linear layer, resnet is 2048 out
        vision_module = torchvision.models.resnet152(
            pretrained=True
        )
        vision_module.fc = torch.nn.Linear(
            in_features=2048,
            out_features=self.vision_feature_dim
        )

        return LanguageAndVisionConcat(
            num_classes=self.hparams.get("num_classes", 2),
            loss_fn=torch.nn.CrossEntropyLoss(),
            language_module=language_module,
            vision_module=vision_module,
            language_feature_dim=self.language_feature_dim,
            vision_feature_dim=self.vision_feature_dim,
            fusion_output_size=self.hparams.get(
                "fusion_output_size", 512
            ),
            dropout_p=self.hparams.get("dropout_p", 0.1),
        )

    def _get_trainer_params(self):
        checkpoint_callback = callbacks.ModelCheckpoint(
            dirpath=self.output_path,
            monitor=self.hparams.get(
                "checkpoint_monitor", "val_loss"
            ),
            mode=self.hparams.get(
                "checkpoint_monitor_mode", "min"
            ),
            verbose=self.hparams.get("verbose", True)
        )

        early_stop_callback = callbacks.EarlyStopping(
            mode='max',
            monitor=self.hparams.get(
                "early_stop_monitor", 'val_loss'
            ),
            min_delta=self.hparams.get(
                "early_stop_min_delta", 0.001
            ),
            patience=self.hparams.get(
                "early_stop_patience", 3
            ),
            verbose=self.hparams.get("verbose", True),
        )

        trainer_params = {
            "checkpoint_callback": True,
            "callbacks": [checkpoint_callback, early_stop_callback],
            "accumulate_grad_batches": self.hparams.get(
                "accumulate_grad_batches", 1
            ),
            "gpus": self.hparams.get("n_gpu", 1),
            "max_epochs": self.hparams.get("max_epochs", 100),
            "gradient_clip_val": self.hparams.get("gradient_clip_value", 1),
        }
        return trainer_params

    @torch.no_grad()
    def make_submission_frame(self, test_path):
        self.hparams.update()


        test_dataset = self._build_dataset(test_path)
        submission_frame = pd.DataFrame(
            index=test_dataset.samples_frame.id,
            columns=["proba", "label"]
        )
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            shuffle=False,
            batch_size=self.hparams.get("batch_size", 4),
            num_workers=self.hparams.get("num_workers", 16))
        for batch in tqdm(test_dataloader, total=len(test_dataloader)):
            preds, _ = self.model.eval().to("cpu")(
                batch["text"], batch["image"]
            )
            submission_frame.loc[batch["id"], "proba"] = preds[:, 1]
            submission_frame.loc[batch["id"], "label"] = preds.argmax(dim=1)
        submission_frame.proba = submission_frame.proba.astype(float)
        submission_frame.label = submission_frame.label.astype(int)
        return submission_frame