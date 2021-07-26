from pathlib import Path
from HatefulMemesModel import HatefulMemesModel
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

data_dir = Path.cwd().parent / "data"
test_unseen_path = data_dir / "test_unseen.jsonl"
test_seen_path = data_dir / "test_seen.jsonl"


def generate_metrics(test_path, tag):
    # make predictions using best-performing checkpoint params:
    checkpoints = list(Path("model-outputs").glob("*.ckpt"))
    assert len(checkpoints) == 1

    ckp = Path.cwd() / checkpoints[0]
    hateful_memes_model = HatefulMemesModel.load_from_checkpoint(checkpoints[0])
    predictions = hateful_memes_model.make_submission_frame(test_path=str(test_path))

    predictions.head()
    predictions.groupby("label").proba.mean()
    predictions.label.value_counts()
    predictions.to_csv(("model-outputs/" + str(checkpoints[0]).split("/")[1].split(".")[0] + tag + ".csv"), index=True)

    # calculate AUROC and accuracy:

    test_df = pd.read_json(test_path, lines=True)
    pred_df = pd.read_csv("model-outputs/" + str(checkpoints[0]).split("/")[1].split(".")[0] + tag + ".csv")
    # pred_df = pd.read_csv("model-outputs/epoch=0-step=94.csv")

    assert len(pred_df) == len(test_df)

    auroc_label = roc_auc_score(test_df["label"], pred_df["label"])
    auroc_proba = roc_auc_score(test_df["label"], pred_df["proba"])
    acc = accuracy_score(test_df["label"], pred_df["label"])
    metrics = {'auroc_label': [auroc_label], 'auroc_proba': [auroc_proba], 'accuracy': [acc]}
    metrics_df = pd.DataFrame(data=metrics)
    metrics_df.to_csv("model-outputs/" + str(checkpoints[0]).split("/")[1].split(".")[0] + tag + "_metrics.csv",
                      index=False)
    # metrics_df.to_csv("model-outputs/epoch=0-step=94_metrics.csv", index=False)

    return


generate_metrics(test_path=test_seen_path, tag="_seen")
generate_metrics(test_path=test_unseen_path, tag="_unseen")

