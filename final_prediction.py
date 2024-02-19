"""
Perform predictions
"""
from decision_tree_random_forest import dt_prediction, rf_prediction
import os
import json
from argparse import ArgumentParser, Namespace
import pandas as pd
import yaml
from tqdm import tqdm
import tensorflow as tf
from bert_classifier import Classifier
import lightning.pytorch as pl
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, RandomSampler
from torchnlp.encoders import LabelEncoder
from torchnlp.utils import collate_tensors, lengths_to_mask
from transformers import AutoModel


def load_model_from_experiment(experiment_folder: str):
    """Function that loads the model from an experiment folder.
    :param experiment_folder: Path to the experiment folder.
    Return:
        - Pretrained model.
    """
    hparams_file = experiment_folder + "/hparams.yaml"
    hparams = yaml.load(open(hparams_file).read(), Loader=yaml.FullLoader)
    with open(os.path.join(experiment_folder,"test.json"),"r") as f:
        testData=json.load(f)

    #checkpoints = [
    #    file
    #    for file in os.listdir(experiment_folder + "/checkpoints/")
    #    if file.endswith(".ckpt")
    #]
    #checkpoint_path = experiment_folder + "/checkpoints/" + checkpoints[-1]
    checkpoint_path=testData['best_model']
    model = Classifier.load_from_checkpoint(
        checkpoint_path, hparams=Namespace(**hparams)
    )
    # Make sure model is in prediction mode
    model.eval()
    model.freeze()
    return model


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Minimalist Transformer Classifier", add_help=True
    )
    parser.add_argument(
        "--experiment",
        required=True,
        type=str,
        help="Path to the experiment folder.",
    )
    parser.add_argument(
        "--input",
        required=True,
        type=str,
        help="Path to the input data.",
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")
    hparams = parser.parse_args()
    print("Loading model...")
    model = load_model_from_experiment(hparams.experiment).to(device)
    print(model)

    testset = pd.read_csv(hparams.input, sep=',', header=0)
    testset = testset.loc[:,["text"]]
    testsetDict = testset.to_dict("records")

    predictions = [
        model.predict(sample)
        for sample in tqdm(testsetDict, desc="Predicting on {}".format(hparams.input))
    ]
    y_pred = [o["predicted_label"] for o in predictions]


    testset["pred"]=y_pred
    alphabetical_emotions = ['anger', 'contempt', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise', ]
    os.makedirs('final_data/predictions', exist_ok=True)
    open('final_data/predictions/answer.txt', 'w')
    dt_pred_list = dt_prediction.tolist()
    dt_pred_list = [alphabetical_emotions[i] for i in dt_pred_list]
    rf_pred_list = rf_prediction.tolist()
    rf_pred_list = [alphabetical_emotions[i] for i in rf_pred_list]

    bert_pred_list = y_pred
    prediction = []

    possible_emotions = ['neutral', 'joy', 'anger', 'contempt', 'sadness', 'fear', 'surprise', 'disgust', ]
    def add_pred(p1, p2, p3):

        frequency = [0, 0, 0, 0, 0, 0, 0, 0]
        def add_freq(x):
            index = possible_emotions.index(x)
            frequency[index] += 1
        add_freq(p1)
        add_freq(p2)
        add_freq(p3)
        if 3 in frequency:
            index = frequency.index(3)
            prediction.append(possible_emotions[index])
        elif 2 in frequency:
            index = frequency.index(2)
            prediction.append(possible_emotions[index])
        else:
            prediction.append(p3)

    for i in range(0, len(dt_pred_list)):
        p1 = dt_pred_list[i]
        p2 = rf_pred_list[i]
        p3 = bert_pred_list[i]
        add_pred(p1, p2, p3)
    for i in range(1581, 17913):
        prediction.append(1.0)
    with open('final_data/predictions/answer.txt', 'w') as outfile:
        outfile.write('\n'.join(str(i) for i in prediction))