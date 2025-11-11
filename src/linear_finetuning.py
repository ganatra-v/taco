import argparse
import numpy as np
import random
import logging
import os
from dataloader import get_split_data, load_image
from torchvision import transforms
from torchvision.models import resnet18, resnet34, resnet50, resnet101, vit_b_16, vit_b_32
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
import torch
from tqdm import tqdm

parser = argparse.ArgumentParser()
# dataset args
parser.add_argument("--dataset", required=True, help="Name of the dataset to use.")
parser.add_argument("--dataset_path", required=True, help="Path to the dataset you want to use.")
parser.add_argument("--fold", type=int, default=0, help="Fold for Cross-Validation")
parser.add_argument("--fold_path", type=str, required=True)
parser.add_argument("--threshold", type=float, default=12.9, help="Threshold for labeling in neojaundice dataset.")

parser.add_argument("--data_prop", type=float, default=0.8, help="Proportion of data used for training the model.")
parser.add_argument("--seed", type=int, default=42, help="Random seed. Default is 42.")
parser.add_argument("--outdir", type=str, required=True)

# model args
parser.add_argument("--arch", choices=["resnet18", "resnet34", "resnet50", "resnet101", "vit_b_16, vit_b_32"], default="resnet18")
parser.add_argument("--projector", action="store_true", help="Whether or not to train a projector")
parser.add_argument("--finetune", type=str, choices=["all", "linear", "proj"], default="all", help="Which parameters to finetune. Default is all.")
parser.add_argument("--pretrained", action="store_true", help="Use pretrained weights. Default is False.")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training. Default is 32.")
parser.add_argument('--weight_decay', type=float, default=0.001, help="L2 regularization")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate. Default is 1e-4.")
parser.add_argument("--epochs", type=int, default=10, help="Number of epochs. Default is 10.")
parser.add_argument("--early_stop", type=int, default=5, help="Early stopping patience. Default is 5.")

def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def configure_output_directory(outdir, args):
    train_details = f"{args.dataset}/linear_finetuning/{args.arch}_batchsize_{args.batch_size}_{args.epochs}_epochs_lr_{args.lr}_seed_{args.seed}_weight_decay_{args.weight_decay}_fold_{args.fold}_data_prop_{args.data_prop}"
    if args.pretrained:
        train_details += "_pretrained"
    if args.projector:
        train_details += "_projector"
    train_details += f"_finetune_{args.finetune}"
    outdir = os.path.join(outdir, train_details)
    os.makedirs(outdir, exist_ok=True)
    return outdir

def load_dataset(args):
    train_data, val_data, test_data = get_split_data(args)
    n_total = len(train_data) + len(val_data) + len(test_data)  
    n_total = n_total // 3 # since there are 3 images per pid
    logging.info(f"Total samples across train, val, test: {n_total}")  
    if args.dataset == "neojaundice":
        train_data["pid"] = train_data["images"].apply(lambda x: x.split("-")[0])
        ref_pids = train_data[train_data["tsb"] == args.threshold]["pid"].unique()
        train_pids = train_data["pid"].unique()
        train_pids = np.random.choice(train_pids, replace=False, size=int(args.data_prop * n_total))
        train_pids = train_pids[:-len(ref_pids)].tolist() + ref_pids.tolist()
        train_data = train_data[train_data["pid"].isin(train_pids)]

    logging.info(f"Trainset = {len(train_data)}, Valset = {len(val_data)}, Testset = {len(test_data)}")
    logging.info(f"Train = {train_data["label"].value_counts()}, Valset = {val_data["label"].value_counts()} Test = {test_data["label"].value_counts()}")        

    train_images = train_data["images"].apply(lambda x: os.path.join(args.dataset_path, x)).values if args.dataset == "neojaundice" else train_data["images"].values
    train_labels = train_data["label"].values    

    val_images = val_data["images"].apply(lambda x: os.path.join(args.dataset_path, x)).values if args.dataset == "neojaundice" else val_data["images"].values
    val_labels = val_data["label"].values

    test_images = test_data["images"].apply(lambda x: os.path.join(args.dataset_path, x)).values if args.dataset == "neojaundice" else test_data["images"].values
    test_labels = test_data["label"].values

    train_dataset = LabelDataset(train_images, train_labels)
    val_dataset = LabelDataset(val_images, val_labels)
    test_dataset = LabelDataset(test_images, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    return train_loader, val_loader, test_loader, train_images, val_images, test_images

class LabelDataset(Dataset):
    def __init__(self, images, labels, split="train"):
        self.images = [load_image(img) for img in tqdm(images)]
        self.labels = labels
        if split == "train":
            self.transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(20)
            ])
        else:
            self.transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.transforms(self.images[idx])
        return image, self.labels[idx]
    

class FinetuneModel(torch.nn.Module):
    def __init__(self, args):
        super(FinetuneModel, self).__init__()
        self.args = args
        if args.arch == "resnet18":
            self.model = resnet18(weights="IMAGENET1K_V1" if args.pretrained else None)
            in_features = self.model.fc.in_features
            self.model.fc = torch.nn.Identity()
        elif args.arch == "resnet34":
            self.model = resnet34(weights="IMAGENET1K_V1" if args.pretrained else None)
            in_features = self.model.fc.in_features
            self.model.fc = torch.nn.Identity()
        elif args.arch == "resnet50":
            self.model = resnet50(weights="IMAGENET1K_V1" if args.pretrained else None)
            in_features = self.model.fc.in_features
            self.model.fc = torch.nn.Identity()
        
        if args.projector:
            self.projector = torch.nn.Sequential(
                torch.nn.Linear(in_features, in_features * 2),
                torch.nn.ReLU(),
                torch.nn.Linear(in_features * 2, in_features)
            )
        self.fc = torch.nn.Linear(in_features, 1)  # assuming binary classification

        if args.finetune == "linear":
            for param in self.model.parameters():
                param.requires_grad = False
            if args.projector:
                for param in self.projector.parameters():
                    param.requires_grad = False
            for param in self.fc.parameters():
                param.requires_grad = True
        elif args.projector and args.finetune == "proj":
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.projector.parameters():
                param.requires_grad = True
            for param in self.fc.parameters():
                param.requires_grad = True
        elif args.finetune == "all":
            for param in self.model.parameters():
                param.requires_grad = True
            if args.projector:
                for param in self.projector.parameters():
                    param.requires_grad = True
            for param in self.fc.parameters():
                param.requires_grad = True
        else:
            raise ValueError("Invalid finetune option")
    
    def forward(self, x):
        features = self.model(x)
        if self.args.projector:
            features = self.projector(features)
        out = self.fc(features)
        return out
    
    def eval_model(self, dataloader):
        self.eval()
        all_outputs = []
        all_labels = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        with torch.no_grad():
            for data in dataloader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = self.forward(inputs)
                outputs = torch.nn.functional.sigmoid(outputs)
                all_outputs.append(outputs.cpu())
                all_labels.append(labels.cpu())
        all_outputs = torch.cat(all_outputs, dim=0).cpu().numpy()
        all_labels = torch.cat(all_labels, dim=0).cpu().numpy()
        return self.calculate_metrics(all_outputs, all_labels)
    
    def calculate_metrics(self, outputs, labels):
        preds = (outputs >= 0.5).astype(int)
        labels = labels.astype(int).flatten()
        accuracy = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds)
        precision = precision_score(labels, preds)
        recall = recall_score(labels, preds)
        sens = recall  # sensitivity is the same as recall
        spec = recall_score(labels, preds, pos_label=0)  # specificity

        logging.info(f"Accuracy: {accuracy}, F1 Score: {f1}, Precision: {precision}, Recall: {recall}, Sensitivity: {sens}, Specificity: {spec}")
        return {
            "accuracy": accuracy,
            "f1_score": f1,
            "precision": precision,
            "recall": recall,
            "sensitivity": sens,
            "specificity": spec
        }

    
    def train_model(self, trainloader, valloader):
        self.train()
        # define optimizer and loss function
        loss_fn = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay
        )
        best_f1 = -1
        best_acc_epoch = 0

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

        for epoch in range(1, self.args.epochs + 1):
            if (epoch - best_acc_epoch) > self.args.early_stop:
                break
            running_loss = 0.0
            for i, data in enumerate(trainloader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                labels = labels.float().unsqueeze(1)

                optimizer.zero_grad()
                outputs = self.forward(inputs)
                outputs = torch.nn.functional.sigmoid(outputs)
                loss = loss_fn(outputs, labels)
                running_loss += loss.item()
                loss.backward()
                optimizer.step()
                logging.info(f"step {i+1}/{len(trainloader)}, loss: {loss.item():.4f}")
            avg_loss = running_loss / len(trainloader)
            logging.info(f"Epoch [{epoch}/{self.args.epochs}], Loss: {avg_loss:.4f}")
            # evaluate on validation set
            metrics = self.eval_model(valloader)
            if metrics["f1_score"] > best_f1:
                best_f1 = metrics["f1_score"]
                best_acc_epoch = epoch
                # save the best model
                torch.save(self.state_dict(), os.path.join(self.args.outdir, "best_model.pth"))
                logging.info(f"Best model saved at epoch {epoch} with F1 Score: {best_f1:.4f}")
        logging.info("Training complete.")
                





if __name__ == "__main__":
    args = parser.parse_args()
 
    # seed everything
    setup_seed(args.seed)

    # configure output directory
    outdir = configure_output_directory(args.outdir, args)
    print(f"Output directory: {outdir}")
    args.outdir = outdir

    print(vars(args))

    logging.basicConfig(
        level=logging.INFO, filename=os.path.join(args.outdir, "log.txt")
    )

    # load dataloaders
    trainloader, valloader, testloader, train_images, val_images, test_images = load_dataset(args)

    # initialize model
    model = FinetuneModel(args)
    # train model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train_model(trainloader, valloader)

    # load best model for evaluation
    model.load_state_dict(torch.load(os.path.join(args.outdir, "best_model.pth")))
    model.eval()
    test_metrics = model.eval_model(testloader)
    logging.info(f"Test Metrics: {test_metrics}")
    logging.info("Finetuning complete.")



