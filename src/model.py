from dataloader import load_image
import logging
import numpy as np
import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
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
        self.projector = nn.Sequential(
            nn.Linear(num_ftrs, num_ftrs * 2),
            nn.ReLU(),
            nn.Linear(num_ftrs * 2, num_ftrs),
            nn.ReLU()
        )
        self.fc = nn.Linear(num_ftrs * 2, 1)

        if args.finetune == "all":
            for param in self.model.parameters():
                param.requires_grad = True
        elif args.finetune == "linear":
            for param in self.model.parameters():
                param.requires_grad = False
        logging.info(f"#-params: {self.get_total_params()}")
        logging.info(f"#-trainable-params: {self.get_trainable_params()}")

    def forward(self, x1, x2):
        x1 = self.projector(self.model(x1))
        x2 = self.projector(self.model(x2))
        x = torch.cat((x1, x2), dim=1)
        return self.fc(x)

    def get_total_params(self):
        return sum([p.numel() for p in self.model.parameters()] + [p.numel() for p in self.projector.parameters()] + [p.numel() for p in self.fc.parameters()]) 

    def get_trainable_params(self):
        return sum([p.numel() for p in self.model.parameters() if p.requires_grad] + [p.numel() for p in self.projector.parameters() if p.requires_grad] + [p.numel() for p in self.fc.parameters() if p.requires_grad])

    def train_model(self, trainloader, infer_trainloaders, reference_images, outdir, infer_val_loaders):
        self.model.train()
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()), lr=self.args.lr, weight_decay = self.args.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers = [torch.optim.lr_scheduler.LinearLR(optimizer, start_factor = 0.1, end_factor = 1.0, total_iters = 10 * len(trainloader)),
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min = 2.5e-6, T_max = (self.args.epochs - 10) * len(trainloader))
        ], milestones = [5 * len(trainloader)])
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min = 1e-6, T_max = self.args.epochs * len(trainloader))

        losses = []
        accs = []
        best_f1 = -1
        best_acc_epoch = 0

        for epoch in range(1, self.args.epochs + 1):
            if (epoch - best_acc_epoch) > args.early_stop:
                break
            self.model.train()
            running_loss = 0.0
            lrs = []
            for i, data in enumerate(trainloader, 0):
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
                accs.append(metrics["acc"])
                for loader, img, val_loader in zip(infer_trainloaders, reference_images, infer_val_loaders):
                    logging.info(f"classification perf. (trainset) - {img}")
                    metrics = self.eval_model(loader)
                    if metrics["f1"] >= best_f1:
                        logging.info("saving best model..................")
                        torch.save(self.state_dict(), os.path.join(outdir, "best_model.pth"))
                        best_f1 = metrics["f1"]
                        best_acc_epoch = epoch
                    logging.info(f"classification perf. (valset) - {img}")
                    metrics = self.eval_model(val_loader)


        logging.info("Finished Training")
        return losses, accs

    def eval_model(self, dataloader, save = False, outfilename = None, val_image_names = None):
        self.model.eval()
        probs_ = []
        preds_ = []
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
                outputs = torch.sigmoid(outputs)
                probs_.append(outputs.cpu().numpy())
                preds = (outputs > 0.5).float()
                preds_.append(preds.cpu().numpy())
                labels_.append(labels.cpu().numpy())

        probs_ = np.vstack(probs_).reshape(-1)
        preds_ = np.vstack(preds_).reshape(-1)
        labels_ = np.vstack(labels_).reshape(-1)

        if save and outfilename is not None:
            df = pd.DataFrame({
                "proba": probs_,
                "label": labels_,
                "image_names": val_image_names
                })
            df.to_csv(outfilename, index = False)

        metrics = self.eval_metrics(preds_, labels_)
        return metrics

    def eval_metrics(self, preds, labels):
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
