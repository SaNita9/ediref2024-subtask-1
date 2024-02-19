"""
Runs a model on a single node across N-gpus.
"""
import argparse
import os
from datetime import datetime

from bert_classifier import Classifier
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import  TensorBoardLogger
from torchnlp.random import set_seed
import json

def main(hparams) -> None:
    """
    Main training routine specific for this project
    :param hparams:
    """
    set_seed(hparams.seed)
    # ------------------------
    # 1 INIT LIGHTNING MODEL AND DATA
    # ------------------------
    model = Classifier(hparams)

    # ------------------------
    # 2 INIT EARLY STOPPING
    # ------------------------
    early_stop_callback = EarlyStopping(
        monitor=hparams.monitor,
        min_delta=0.0,
        patience=hparams.patience,
        verbose=True,
        mode=hparams.metric_mode,
    )

    # ------------------------
    # 3 INIT LOGGERS
    # ------------------------
    # Tensorboard Callback
    tb_logger = TensorBoardLogger(
        save_dir=hparams.out,
        version=".",
        name="",
    )

    # Model Checkpoint Callback
    ckpt_path = os.path.join(
        hparams.out,
        tb_logger.version,
        "checkpoints",
    )

    # --------------------------------
    # 4 INIT MODEL CHECKPOINT CALLBACK
    # -------------------------------
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_path,
        save_top_k=hparams.save_top_k,
        verbose=True,
        monitor=hparams.monitor,
        mode=hparams.metric_mode,
        save_weights_only=True,
    )

    # ------------------------
    # 5 INIT TRAINER
    # ------------------------
    trainer = Trainer(
        logger=tb_logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        gradient_clip_val=1.0,
        devices=hparams.gpus,
        deterministic=True,
        check_val_every_n_epoch=1,
        fast_dev_run=False,
        accumulate_grad_batches=hparams.accumulate_grad_batches,
        max_epochs=hparams.max_epochs,
        min_epochs=hparams.min_epochs,
        val_check_interval=hparams.val_check_interval,
        #distributed_backend="dp",
    )
    # ------------------------
    # 6 START TRAINING
    # ------------------------
    train_r=trainer.fit(model, model.data)
    print(train_r)

    test_r=trainer.test(ckpt_path="best", datamodule=model.data)
    print(test_r)

    with open(os.path.join(hparams.out,"params.json"),"w") as f:
        json.dump(vars(hparams),f,indent=4)

    data={
        'best_model':checkpoint_callback.best_model_path,
        'accuracy':test_r[0]['test_acc'],
        'macro avg':{
            'f1-score':test_r[0]['test_f1macro']
        },
        'weighted avg':{
            'f1-score':test_r[0]['test_f1weighted']
        },
    }
    with open(os.path.join(hparams.out,"test.json"),"w") as f:
        json.dump(data,f,indent=4)


if __name__ == "__main__":
    # ------------------------
    # TRAINING ARGUMENTS
    # ------------------------
    # these are project-wide arguments
    parser = argparse.ArgumentParser(
        description="Minimalist Transformer Classifier",
        add_help=True,
    )
    parser.add_argument("--seed", type=int, default=3, help="Training seed.")
    parser.add_argument(
        "--save_top_k",
        default=1,
        type=int,
        help="The best k models according to the quantity monitored will be saved.",
    )
    # Early Stopping
    parser.add_argument(
        "--monitor", default="val_f1weighted", type=str, help="Quantity to monitor." #####era val_acc
    )
    parser.add_argument(
        "--metric_mode",
        default="max",
        type=str,
        help="If we want to min/max the monitored quantity.",
        choices=["auto", "min", "max"],
    )
    parser.add_argument(
        "--patience",
        default=2,
        type=int,
        help=(
            "Number of epochs with no improvement "
            "after which training will be stopped."
        ),
    )
    parser.add_argument(
        "--min_epochs",
        default=1,
        type=int,
        help="Limits training to a minimum number of epochs",
    )
    parser.add_argument(
        "--max_epochs",
        default=5,
        type=int,
        help="Limits training to a max number number of epochs",
    )

    # Batching
    parser.add_argument(
        "--batch_size", default=6, type=int, help="Batch size to be used."
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        default=2,
        type=int,
        help=(
            "Accumulated gradients runs K small batches of size N before "
            "doing a backwards pass."
        ),
    )

    # gpu args
    parser.add_argument("--gpus", type=int, default=1, help="How many gpus")
    parser.add_argument(
        "--val_check_interval",
        default=1.0,
        type=float,
        help=(
            "If you don't want to use the entire dev set (for debugging or "
            "if it's huge), set how much of the dev set you want to use with this flag."
        ),
    )


    parser.add_argument("--out", required=True, help="Save path")


    # each LightningModule defines arguments relevant to it
    parser = Classifier.add_model_specific_args(parser)
    hparams = parser.parse_args()

    # ---------------------
    # RUN TRAINING
    # ---------------------
    main(hparams)
