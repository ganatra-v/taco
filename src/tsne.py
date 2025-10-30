from sklearn.manifold import TSNE
import os
import numpy as np
from model import taco
import torch
from tqdm import tqdm
from dataloader import load_image
from torchvision import transforms
import matplotlib.pyplot as plt
import seaborn as sns

path = <PATH>
model_path = path + "best_model.pth"
data_path = path + "all_data.txt"

with open(data_path, "r") as f:
    data = f.readlines()

with open(path + "val_data.txt", "r") as f:
    val_data = f.readlines()

val_data = [f.strip().split(",")[0] for f in val_data]


images = []
hgb = []

for line in data:
    line = line.strip()
    img, value = line.split(",")
    # if img not in val_data:
    images.append(img)
    hgb.append(float(value))

class Args:
    arch = "resnet34"
    finetune = "linear"
    pretrained = True

print(Args.finetune)
model = taco(Args)
for p in model.parameters():
    print(p[0])
    break
# model.load_state_dict(torch.load(model_path))
for p in model.parameters():
    print(p[0])
    break
model.cuda()
model.eval()

mean_ =[0.485, 0.456, 0.406]
std_ = [0.229, 0.224, 0.225]

transforms_ = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean= mean_ , std=std_),
])

embeddings = []
if os.path.exists(path + "all_embeddings.npy"):
    embeddings = np.load(path + "all_embeddings.npy")
else:
    for img in tqdm(images):
        with torch.no_grad():
            img = load_image(img)
            img = transforms_(img).unsqueeze(0).cuda()
            embedding = model.model(img)
            embeddings.append(embedding.cpu().detach().numpy())

embeddings = np.vstack(embeddings)
print(embeddings.shape)
np.save(path + "all_embeddings.npy", embeddings)

for i in range(3, 20):
    tsne = TSNE(n_components = 2, perplexity = i)
    x_ = tsne.fit_transform(embeddings)

    plt.scatter(x_[:,1], x_[:, 0], c = hgb, cmap = "viridis")
    plt.colorbar()
    plt.savefig(f"./tsne_{i}.png", bbox_inches = "tight")
    plt.close()