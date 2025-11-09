from dataloader import load_image
import logging
import numpy as np
import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
import torch
import torch.nn as nn
from torchvision.models import (
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    resnet152,
    vit_b_16,
    vit_b_32,
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    ResNet101_Weights,
    ResNet152_Weights,
    ViT_B_16_Weights,
    ViT_B_32_Weights,
)
from torchvision import transforms
from tqdm import tqdm


class taco(nn.Module):
    def __init__(self, args):
        super(taco, self).__init__()
        self.args = args
        arch = args.arch
        if arch == "resnet18":
            self.model = (
                resnet18(weights=ResNet18_Weights.DEFAULT)
                if args.pretrained
                else resnet18(weights=None)
            )
            num_ftrs = self.model.fc.in_features
        elif arch == "resnet34":
            self.model = (
                resnet34(weights=ResNet34_Weights.DEFAULT)
                if args.pretrained
                else resnet34(weights=None)
            )
            num_ftrs = self.model.fc.in_features
        elif arch == "resnet50":
            self.model = (
                resnet50(weights=ResNet50_Weights.DEFAULT)
                if args.pretrained
                else resnet50(weights=None)
            )
            num_ftrs = self.model.fc.in_features
        elif arch == "resnet101":
            self.model = (
                resnet101(weights=ResNet101_Weights.DEFAULT)
                if args.pretrained
                else resnet101(weights=None)
            )
            num_ftrs = self.model.fc.in_features
        elif arch == "resnet152":
            self.model = (
                resnet152(weights=ResNet152_Weights.DEFAULT)
                if args.pretrained
                else resnet152(weights=None)
            )
            num_ftrs = self.model.fc.in_features
        elif arch == "vit_b_16":
            self.model = (
                vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
                if args.pretrained
                else vit_b_16(weights=None)
            )
            num_ftrs = self.model.heads.head.in_features
        elif arch == "vit_b_32":
            self.model = (
                vit_b_32(weights=ViT_B_32_Weights.DEFAULT)
                if args.pretrained
                else vit_b_32(weights=None)
            )
            num_ftrs = self.model.heads.head.in_features

        self.model.fc = nn.Identity()
        if self.args.projector:
            self.projector = nn.Sequential(
                nn.Linear(num_ftrs, num_ftrs * 2),
                nn.ReLU(),
                nn.Linear(num_ftrs * 2, num_ftrs),
                nn.ReLU()
            )
        self.fc = nn.Linear(num_ftrs * 2, 1)

        if args.finetune == "proj" and args.projector:
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.projector.parameters():
                param.requires_grad = True
            for param in self.fc.parameters():
                param.requires_grad = True
        elif args.finetune == "linear":
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.fc.parameters():
                param.requires_grad = True
            if args.projector:
                for param in self.projector.parameters():
                    param.requires_grad = False
        elif args.finetune == "all":
            for param in self.model.parameters():
                param.requires_grad = True
            if args.projector:
                for param in self.projector.parameters():
                    param.requires_grad = True
            for param in self.fc.parameters():
                param.requires_grad = True

        logging.info(f"#-params: {self.get_total_params()}")
        logging.info(f"#-trainable-params: {self.get_trainable_params()}")

    def forward(self, x1, x2):
        x1 = self.model(x1)        
        x2 = self.model(x2)

        if self.args.projector:
            x1 = self.projector(x1)
            x2 = self.projector(x2)

        x = torch.cat((x1, x2), dim=1)
        return self.fc(x)

    def get_total_params(self):
        if self.args.projector:
            return sum([p.numel() for p in self.model.parameters()] + [p.numel() for p in self.projector.parameters()] + [p.numel() for p in self.fc.parameters()])
        else:
            return sum([p.numel() for p in self.model.parameters()] + [p.numel() for p in self.fc.parameters()])

    def get_trainable_params(self):
        if self.args.projector:
            return sum([p.numel() for p in self.model.parameters() if p.requires_grad] + [p.numel() for p in self.projector.parameters() if p.requires_grad] + [p.numel() for p in self.fc.parameters() if p.requires_grad])
        else:
            return sum([p.numel() for p in self.model.parameters() if p.requires_grad] + [p.numel() for p in self.fc.parameters() if p.requires_grad])

    def train_model(self, trainloader, infer_val_loaders, reference_images, outdir, scaler = None):
        print("Starting training...")
        if self.args.task == "reg":
            criterion = nn.MSELoss()
        else:
            criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()), lr=self.args.lr, weight_decay = self.args.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers = [torch.optim.lr_scheduler.LinearLR(optimizer, start_factor = 0.1, end_factor = 1.0, total_iters = 5 * len(trainloader)),
            torch.optim.lr_scheduler.MultiStepLR(optimizer, [(self.args.epochs - 10) * len(trainloader)], gamma = 0.1)
        ], milestones = [5 * len(trainloader)])

        losses = []
        accs = []
        best_f1 = -1
        best_acc_epoch = 0

        for epoch in range(1, self.args.epochs + 1):
            if (epoch - best_acc_epoch) > self.args.early_stop:
                break
            self.model.train()
            running_loss = 0.0
            lrs = []
            for i, data in enumerate(trainloader):
                inputs1, inputs2, labels = data
                inputs1, inputs2, labels = (
                    (
                        inputs1.cuda(),
                        inputs2.cuda(),
                        labels.float().unsqueeze(1).cuda(),
                    )
                    if torch.cuda.is_available()
                    else (inputs1, inputs2, labels.float().unsqueeze(1))
                )

                optimizer.zero_grad()

                outputs = self.forward(inputs1, inputs2)
                if "cls" in self.args.task:
                    outputs = torch.sigmoid(outputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                lrs = scheduler.get_last_lr()[0]

                scheduler.step()
            logging.info(f"epoch {epoch}, lrs={lrs}, loss: {running_loss / (i+1)}")
            losses.append(running_loss / (i + 1))

            if epoch % 1 == 0:
                logging.info(f"comparison perf. (trainset)")
                metrics = self.eval_model(trainloader)
                if "cls" in self.args.task:
                    accs.append(metrics["acc"])
                elif "reg" in self.args.task:
                    accs.append(metrics["mae"]) 

                for loader, img in zip(infer_val_loaders, reference_images):
                    logging.info(f"classification perf. (valset) - {img}")
                    metrics = self.eval_model(loader, scaler = scaler)
                    logging.info(metrics)
                    if "cls" in self.args.task and metrics["f1"] >= best_f1:
                        logging.info("saving best model..................")
                        torch.save(self.state_dict(), os.path.join(outdir, "best_model.pth"))
                        best_f1 = metrics["f1"]
                        best_acc_epoch = epoch
                    elif "reg" in self.args.task and metrics["mae"] >= best_f1:
                        logging.info("saving best model..................")
                        torch.save(self.state_dict(), os.path.join(outdir, "best_model.pth"))
                        best_f1 = metrics["mae"]
                        best_acc_epoch = epoch

        logging.info("Finished Training")
        return losses, accs

    def eval_model(self, dataloader, save = False, outfilename = None, val_image_names = None, scaler = None):
        self.model.eval()
        probs_ = []
        labels_ = []
        with torch.no_grad():
            for data in dataloader:
                inputs1, inputs2, labels = data
                inputs1, inputs2, labels = (
                    (
                        inputs1.cuda(),
                        inputs2.cuda(),
                        labels.float().unsqueeze(1).cuda(),
                    )
                    if torch.cuda.is_available()
                    else (inputs1, inputs2, labels.float().unsqueeze(1))
                )
                outputs = self.forward(inputs1, inputs2)
                if "cls" in self.args.task:
                    outputs = torch.sigmoid(outputs).cpu().numpy()
                if scaler is not None and "reg" in self.args.task:
                    outputs = scaler.inverse_transform(outputs.cpu().numpy())
                probs_.append(outputs)
                labels_.append(labels.cpu().numpy())

        probs_ = np.vstack(probs_).reshape(-1)
        labels_ = np.vstack(labels_).reshape(-1)

        if save and outfilename is not None:
            df = pd.DataFrame({
                "pred": probs_,
                "label": labels_,
                "image_names": val_image_names
                })
            df.to_csv(outfilename, index = False)

        metrics = self.eval_metrics(probs_, labels_)
        return metrics

    def eval_metrics(self, preds, labels):
        if "cls" in self.args.task:
            preds = [1 if p >= 0.5 else 0 for p in preds]
            acc = accuracy_score(labels, preds)
            precision = precision_score(labels, preds)
            recall = recall_score(labels, preds)
            f1 = f1_score(labels, preds)

            sens = recall
            spec = recall_score(labels, preds, pos_label = 0)
            logging.info(f"Acc: {acc}, Prec: {precision}, Recall: {recall}, F1: {f1}, Sens: {sens}, Spec: {spec}")
            return {
                "acc": acc,
                "prec" : precision,
                "rec" : recall,
                "f1" : f1,
                "sens" : sens,
                "spec" : spec
            }
        elif "reg" in self.args.task:
            mae = mean_absolute_error(labels, preds)
            mse = mean_squared_error(labels, preds)
            mape = mean_absolute_percentage_error(labels, preds)
            r2 = r2_score(labels, preds)

            logging.info(f"MAE: {mae}, MSE: {mse}, MAPE: {mape}, R2: {r2}")
            return {
                "mae": mae,
                "mse": mse,
                "mape": mape,
                "r2": r2,
            }