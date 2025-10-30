import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score

filepath = <PREDICTION.csv>
data = pd.read_csv(filepath)

print(data.head())

for threshold in np.arange(0.005, 0.1, 0.005):
    probs = data["proba"].values
    preds = (probs>threshold).astype(int)
    labels = data["label"].values

    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds)
    sens = recall_score(labels, preds)
    spec = recall_score(labels, preds, pos_label=0)
    f1 = f1_score(labels, preds)

    f = 2*sens*spec/(sens + spec)

    print(f"threshold: {threshold:.3f}, acc: {acc:.3f}, sens: {sens:.3f}, spec: {spec:.3f}, f: {f:.3f} prec: {prec:.3f}, recall: {sens:.3f}, f1: {f1:.3f}")