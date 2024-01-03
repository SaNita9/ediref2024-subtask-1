# -*- coding: utf-8 -*-
import logging as log
from argparse import ArgumentParser, Namespace
from collections import OrderedDict

import numpy as np
import pandas as pd
import lightning.pytorch as pl
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, RandomSampler
from torchnlp.encoders import LabelEncoder
from torchnlp.utils import collate_tensors, lengths_to_mask
from transformers import AutoModel

from tokenizer import Tokenizer
from utils import mask_fill

from torchmetrics import Accuracy
from torchmetrics.classification import MulticlassF1Score


class Classifier(pl.LightningModule):
    """
    Sample model to show how to use a Transformer model to classify sentences.

    :param hparams: ArgumentParser containing the hyperparameters.
    """

    class DataModule(pl.LightningDataModule):
        def __init__(self, classifier_instance):
            super().__init__()
            #self.hparams = classifier_instance.hparams
            for key in classifier_instance.hparams.keys():
                self.hparams[key]=classifier_instance.hparams[key]

            self.classifier = classifier_instance
            # Label Encoder
            self.label_encoder = LabelEncoder(
                pd.read_csv(self.hparams.train_csv, sep=',', header=0).label.astype(str).unique().tolist(),
                reserved_labels=[],
            )
            self.label_encoder.unknown_index = None

        def read_csv(self, path: str) -> list:
            """Reads a comma separated value file.

            :param path: path to a csv file.

            :return: List of records as dictionaries
            """
            df = pd.read_csv(path, sep='\t', header=0)
            df = df[["text", "label"]]
            df["text"] = df["text"].astype(str)
            df["label"] = df["label"].astype(str)
            return df.to_dict("records")

        def train_dataloader(self) -> DataLoader:
            """Function that loads the train set."""
            self._train_dataset = self.read_csv(self.hparams.train_csv)
            return DataLoader(
                dataset=self._train_dataset,
                sampler=RandomSampler(self._train_dataset),
                batch_size=self.hparams.batch_size,
                collate_fn=self.classifier.prepare_sample,
                num_workers=self.hparams.loader_workers,
            )

        def val_dataloader(self) -> DataLoader:
            """Function that loads the validation set."""
            self._dev_dataset = self.read_csv(self.hparams.dev_csv)
            return DataLoader(
                dataset=self._dev_dataset,
                batch_size=self.hparams.batch_size,
                collate_fn=self.classifier.prepare_sample,
                num_workers=self.hparams.loader_workers,
            )

        def test_dataloader(self) -> DataLoader:
            """Function that loads the test set."""
            self._test_dataset = self.read_csv(self.hparams.test_csv)
            return DataLoader(
                dataset=self._test_dataset,
                batch_size=self.hparams.batch_size,
                collate_fn=self.classifier.prepare_sample,
                num_workers=self.hparams.loader_workers,
            )

    def __init__(self, hparams: Namespace) -> None:
        super(Classifier, self).__init__()
        #self.hparams = hparams
        hparamsv=vars(hparams)
        for key in hparamsv.keys():
            self.hparams[key]=hparamsv[key]

        self.batch_size = hparams.batch_size

        self.accuracy=Accuracy(task='multiclass',num_classes=2)
        self.f1macro=MulticlassF1Score(num_classes=2, average='macro')
        self.f1weighted=MulticlassF1Score(num_classes=2, average='weighted')

        # Build Data module
        self.data = self.DataModule(self)

        # build model
        self.__build_model()

        # Loss criterion initialization.
        self.__build_loss()

        if hparams.nr_frozen_epochs > 0:
            self.freeze_encoder()
        else:
            self._frozen = False
        self.nr_frozen_epochs = hparams.nr_frozen_epochs

        self.save_hyperparameters()

    def __build_model(self) -> None:
        """Init BERT model + tokenizer + classification head."""
        self.bert = AutoModel.from_pretrained(
            self.hparams.encoder_model, output_hidden_states=True
        )

        # set the number of features our encoder model will return...
        self.encoder_features = self.bert.config.hidden_size

        # Tokenizer
        self.tokenizer = Tokenizer(self.hparams.encoder_model)

        # Classification head
        self.classification_head = nn.Sequential(
            nn.Linear(self.encoder_features, self.encoder_features * 2),
            nn.Tanh(),
            nn.Linear(self.encoder_features * 2, self.encoder_features),
            nn.Tanh(),
            nn.Linear(self.encoder_features, self.data.label_encoder.vocab_size),
        )

    def __build_loss(self):
        """Initializes the loss function/s."""
        self._loss = nn.CrossEntropyLoss()

    def unfreeze_encoder(self) -> None:
        """un-freezes the encoder layer."""
        if self._frozen:
            log.info(f"\n-- Encoder model fine-tuning")
            for param in self.bert.parameters():
                param.requires_grad = True
            self._frozen = False

    def freeze_encoder(self) -> None:
        """freezes the encoder layer."""
        for param in self.bert.parameters():
            param.requires_grad = False
        self._frozen = True

    def predict(self, sample: dict) -> dict:
        """Predict function.
        :param sample: dictionary with the text we want to classify.

        Returns:
            Dictionary with the input text and the predicted label.
        """
        if self.training:
            self.eval()

        with torch.no_grad():
            model_input, _ = self.prepare_sample([sample], prepare_target=False)
            model_out = self.forward(**model_input)
            logits = model_out["logits"].numpy()
            predicted_labels = [
                self.data.label_encoder.index_to_token[prediction]
                for prediction in np.argmax(logits, axis=1)
            ]
            sample["predicted_label"] = predicted_labels[0]

        return sample

    def forward(self, tokens, lengths):
        """Usual pytorch forward function.
        :param tokens: text sequences [batch_size x src_seq_len]
        :param lengths: source lengths [batch_size]

        Returns:
            Dictionary with model outputs (e.g: logits)
        """
        tokens = tokens[:, : lengths.max()]
        # When using just one GPU this should not change behavior
        # but when splitting batches across GPU the tokens have padding
        # from the entire original batch
        mask = lengths_to_mask(lengths, device=tokens.device)

        # Run BERT model.
        word_embeddings = self.bert(tokens, mask)[0]
        
        # Average Pooling
        word_embeddings = mask_fill(
            0.0, tokens, word_embeddings, self.tokenizer.padding_index
        )
        sentemb = torch.sum(word_embeddings, 1)
        sum_mask = mask.unsqueeze(-1).expand(word_embeddings.size()).float().sum(1)
        sentemb = sentemb / sum_mask
        
        return {"logits": self.classification_head(sentemb)}

    def loss(self, predictions: dict, targets: dict) -> torch.tensor:
        """
        Computes Loss value according to a loss function.
        :param predictions: model specific output. Must contain a key 'logits' with
            a tensor [batch_size x 1] with model predictions
        :param labels: Label values [batch_size]

        Returns:
            torch.tensor with loss value.
        """
        return self._loss(predictions["logits"], targets["labels"])

    def prepare_sample(self, sample: list, prepare_target: bool = True) -> (dict, dict):
        """
        Function that prepares a sample to input the model.
        :param sample: list of dictionaries.

        Returns:
            - dictionary with the expected model inputs.
            - dictionary with the expected target labels.
        """
        sample = collate_tensors(sample)
        tokens, lengths = self.tokenizer.batch_encode(sample["text"])

        inputs = {"tokens": tokens, "lengths": lengths}

        if not prepare_target:
            return inputs, {}

        # Prepare target:
        try:
            targets = {"labels": self.data.label_encoder.batch_encode(sample["label"])}
            return inputs, targets
        except RuntimeError:
            raise Exception("Label encoder found an unknown label.")

    def training_step(self, batch: tuple, batch_nb: int, *args, **kwargs) -> dict:
        """
        Runs one training step. This usually consists in the forward function followed
            by the loss function.

        :param batch: The output of your dataloader.
        :param batch_nb: Integer displaying which batch this is

        Returns:
            - dictionary containing the loss and the metrics to be added to the lightning logger.
        """
        inputs, targets = batch
        model_out = self.forward(**inputs)
        loss_val = self.loss(model_out, targets)

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        #if self.trainer.use_dp or self.trainer.use_ddp2:
            #loss_val = loss_val.unsqueeze(0)

        output = OrderedDict({"loss": loss_val})
        # can also return just a scalar instead of a dict (return loss_val)
        return output

    def shared_val_step(self, batch, batch_idx, prefix):
        inputs, targets = batch
        model_out = self.forward(**inputs)
        val_loss = self.loss(model_out, targets)

        y = targets["labels"]
        y_hat = model_out["logits"]
        labels_hat = torch.argmax(y_hat, dim=1)
        self.accuracy(labels_hat, y)
        self.f1macro(labels_hat, y)
        self.f1weighted(labels_hat, y)
        self.log("{prefix}_acc".format(prefix=prefix),self.accuracy, on_epoch=True)
        self.log("{prefix}_f1macro".format(prefix=prefix),self.f1macro, on_epoch=True)
        self.log("{prefix}_f1weighted".format(prefix=prefix),self.f1weighted, on_epoch=True)
        self.log("{prefix}_loss".format(prefix=prefix),val_loss, on_epoch=True)

    def validation_step(self, batch: tuple, batch_nb: int, *args, **kwargs) -> dict:
        """Similar to the training step but with the model in eval mode.

        Returns:
            - dictionary passed to the validation_end function.
        """
        self.shared_val_step(batch, batch_nb,"val")


    def test_step(self, batch: tuple, batch_nb: int, *args, **kwargs) -> dict:
        """Similar to the training step but with the model in eval mode.

        Returns:
            - dictionary passed to the validation_end function.
        """
        self.shared_val_step(batch, batch_nb,"test")


    def configure_optimizers(self):
        """Sets different Learning rates for different parameter groups."""
        parameters = [
            {"params": self.classification_head.parameters()},
            {
                "params": self.bert.parameters(),
                "lr": self.hparams.encoder_learning_rate,
            },
        ]
        optimizer = optim.Adam(parameters, lr=self.hparams.learning_rate)
        return [optimizer], []

    def on_epoch_end(self):
        """Pytorch lightning hook"""
        if self.current_epoch + 1 >= self.nr_frozen_epochs:
            self.unfreeze_encoder()

    @classmethod
    def add_model_specific_args(cls, parser: ArgumentParser) -> ArgumentParser:
        """Parser for Estimator specific arguments/hyperparameters.
        :param parser: argparse.ArgumentParser

        Returns:
            - updated parser
        """
        parser.add_argument(
            "--encoder_model",
            default="bert-base-uncased",
            type=str,
            help="Encoder model to be used.",
        )
        parser.add_argument(
            "--encoder_learning_rate",
            default=1e-05,
            type=float,
            help="Encoder specific learning rate.",
        )
        parser.add_argument(
            "--learning_rate",
            default=3e-05,
            type=float,
            help="Classification head learning rate.",
        )
        parser.add_argument(
            "--nr_frozen_epochs",
            default=1,
            type=int,
            help="Number of epochs we want to keep the encoder model frozen.",
        )
        parser.add_argument(
            "--train_csv",
            default="data/imdb_reviews_train.csv",
            type=str,
            help="Path to the file containing the train data.",
        )
        parser.add_argument(
            "--dev_csv",
            default="data/imdb_reviews_test.csv",
            type=str,
            help="Path to the file containing the dev data.",
        )
        parser.add_argument(
            "--test_csv",
            default="data/imdb_reviews_test.csv",
            type=str,
            help="Path to the file containing the dev data.",
        )
        parser.add_argument(
            "--loader_workers",
            default=8,
            type=int,
            help="How many subprocesses to use for data loading. 0 means that \
                the data will be loaded in the main process.",
        )
        return parser
