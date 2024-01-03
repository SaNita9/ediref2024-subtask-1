"""
Perform predictions
"""
import os
import json
from argparse import ArgumentParser, Namespace

import pandas as pd
import yaml
from sklearn.metrics import classification_report
from tqdm import tqdm

from classifier import Classifier


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
    parser.add_argument(
        "--output",
        required=False,
        type=str,
        help="Path to store predictions.",
    )
    hparams = parser.parse_args()
    print("Loading model...")
    model = load_model_from_experiment(hparams.experiment)
    # print(model)

    testset = pd.read_csv(hparams.input, sep='\t', header=0)
    testset = testset.loc[:,["text"]]
    testsetDict = testset.to_dict("records")
    predictions = [
        model.predict(sample)
        for sample in tqdm(testsetDict, desc="Predicting on {}".format(hparams.input))
    ]
    y_pred = [o["predicted_label"] for o in predictions]
    #y_true = [s["label"] for s in testset]
    #print(classification_report(y_true, y_pred))
    print ("saving predictions at: {}".format(hparams.output))

    testset["pred"]=y_pred
    testset.to_csv(hparams.output, sep="\t", index=False)

    #with open(hparams.output, 'w', encoding='utf-8') as f:
    #    json.dump(predictions, f, ensure_ascii=False, indent=4)

