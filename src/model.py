from dataloader import load_image
import logging
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
        self.fc = nn.Linear(num_ftrs * 2, 1)

        if args.finetune == "all":
            for param in self.model.parameters():
                param.requires_grad = True
        elif args.finetune == "linear":
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, x1, x2):
        x1 = self.model(x1)
        x2 = self.model(x2)
        x = torch.cat((x1, x2), dim=1)
        return self.fc(x)

    def get_total_params(self):
        return sum([p.numel() for p in self.model.parameters()])

    def get_trainable_params(self):
        return sum([p.numel() for p in self.model.parameters() if p.requires_grad])

    def train_model(self, trainloader):
        self.model.train()
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()), lr=self.args.lr
        )

        losses = []
        accs = []

        for epoch in range(1, self.args.epochs + 1):
            running_loss = 0.0
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
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            logging.info(f"epoch {epoch}, loss: {running_loss / (i+1)}")
            losses.append(running_loss / (i + 1))
            acc = self.eval_comparison(trainloader)
            accs.append(acc)
        logging.info("Finished Training")
        return losses, accs

    def eval_comparison(self, dataloader):
        self.model.eval()
        correct = 0
        total = 0
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
                preds = (outputs > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.numel()

        accuracy = correct / total if total > 0 else 0
        logging.info(f"eval acc: {accuracy:.4f}")
        return accuracy

    def eval_model(self, valdataloader):
        self.model.eval()
        reference_image_names = []
        reference_images = []
        with open(f"{self.args.outdir}/reference_images.txt", "r") as f:
            lines = f.readlines()
            for line in lines:
                img_path = line.strip()
                img = load_image(img_path)
                reference_images.append(img)
                reference_image_names.append(img_path)

        val_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=valdataloader.dataset.mean_, std=valdataloader.dataset.std_
                ),
            ]
        )

        ref_vs_acc = {}

        for idx, img in enumerate(reference_images):
            img = val_transforms(img)
            img = img.unsqueeze(0)
            img = img.cuda() if torch.cuda.is_available() else img

            correct = 0
            total = 0

            with torch.no_grad():
                for data in valdataloader:
                    inputs1, _, labels = data
                    inputs2 = img.repeat(inputs1.size(0), 1, 1, 1)
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
                    preds = (outputs > 0.5).float()
                    correct += (preds == labels).sum().item()
                    total += labels.numel()
            accuracy = correct / total if total > 0 else 0
            logging.info(
                f"ref_images: {reference_image_names[idx]}, eval acc: {accuracy:.4f}"
            )
            ref_vs_acc[reference_image_names[idx]] = accuracy
        return ref_vs_acc
