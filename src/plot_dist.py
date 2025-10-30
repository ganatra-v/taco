import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

filepath = <PREDICTION.csv>
data = pd.read_csv(filepath)

probs = data["proba"]
labels = data["label"]

pos = probs[labels == 1]
neg = probs[labels == 0]
print(pos.max(), pos.min(), neg.max(), neg.min())
print(len(pos), len(neg))


sns.kdeplot(pos, label = "Anemic", fill = True)
sns.kdeplot(neg, label = "Healthy", fill = True)

plt.xlabel("probability")
plt.grid()
plt.ylabel("counts")
plt.legend()
plt.savefig("./dist_plots.png", bbox_inches="tight")