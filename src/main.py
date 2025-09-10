import argparse
from dataloader import load_dataset
import json
import logging
from model import taco
import numpy as np
import os
import random
import torch

parser = argparse.ArgumentParser()

parser.add_argument("--dataset", choices=["eyes-defy-anemia", "neojaundice"])
parser.add_argument(
    "--dataset_path", required=True, help="Path to the dataset you want to use."
)
parser.add_argument(
    "--anemia_threshold",
    type=float,
    default=11.5,
    help="Threshold for anemia in g/dL. Default is 11.5 g/dL.",
)


parser.add_argument(
    "--n_comparisons_per_image",
    type=int,
    default=5,
    help="Number of comparisons to generate per image. Default is 5.",
)
parser.add_argument(
    "--outdir",
    type=str,
    required=True,
)
parser.add_argument("--seed", type=int, default=42, help="Random seed. Default is 42.")

parser.add_argument(
    "--arch",
    choices=["resnet18", "resnet34", "resnet50", "resnet101", "vit_b_16, vit_b_32"],
    default="resnet18",
)

parser.add_argument(
    "--finetune",
    type=str,
    choices=["all", "linear"],
    default="all",
    help="Which parameters to finetune. Default is all.",
)

parser.add_argument(
    "--pretrained",
    action="store_true",
    help="Use pretrained weights. Default is False.",
)

parser.add_argument(
    "--batch_size", type=int, default=32, help="Batch size for training. Default is 32."
)

parser.add_argument(
    "--lr", type=float, default=1e-4, help="Learning rate. Default is 1e-4."
)

parser.add_argument(
    "--epochs", type=int, default=10, help="Number of epochs. Default is 10."
)


def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def configure_output_directory(outdir):
    out_dir = os.path.join("outputs", outdir)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


if __name__ == "__main__":
    args = parser.parse_args()
    args.outdir = configure_output_directory(args.outdir)

    print(vars(args))
    logging.basicConfig(
        level=logging.INFO, filename=os.path.join(args.outdir, "log.txt")
    )

    setup_seed(args.seed)

    train_loader, val_loader = load_dataset(args)
    model = taco(args)
    model = model.cuda() if torch.cuda.is_available() else model
    trainlosses, trainaccs = model.train_model(train_loader)

    with open(os.path.join(args.outdir, "train_losses.txt"), "w") as f:
        for loss in trainlosses:
            f.write(f"{loss}\n")
    with open(os.path.join(args.outdir, "train_accs.txt"), "w") as f:
        for acc in trainaccs:
            f.write(f"{acc}\n")

    # save the model
    torch.save(model.state_dict(), os.path.join(args.outdir, "model.pth"))
    logging.info("Model saved to %s", os.path.join(args.outdir, "model.pth"))

    val_acc = model.eval_model(val_loader)
    logging.info(f"Validation accuracy: {val_acc}")
    with open(os.path.join(args.outdir, "val_acc.json"), "w") as f:
        json.dump(val_acc, f)
