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

# dataset args
parser.add_argument("--dataset", required=True, help="Name of the dataset to use.")
parser.add_argument("--dataset_path", required=True, help="Path to the dataset you want to use.")
parser.add_argument("--threshold", type=float, required=True, help="Threshold for diseases. Required for classification tasks.")
parser.add_argument("--data_prop", type=float, default=0.8, help="Proportion of data used for training the model.")
parser.add_argument("--fold", type=int, default=0, help="Fold for Cross-Validation")
parser.add_argument("--fold_path", type=str, required=True)

# method args
parser.add_argument("--n_comparisons_per_image", type=int, default=5, help="Number of comparisons to generate per image. Default is 5.")
parser.add_argument("--comparison_type", choices=["online", "offline"], default="offline", help="Whether to sample comparisons online for each epoch, or offline")
parser.add_argument("--task", choices=["cls", "reg", "repr_cls", "repr_reg"], required=True, help = "Whether to train a model for classification, regression, or representation learning --> Classification and regression will involve output predictions for all subsets, representation learning will not")
parser.add_argument("--ref_image_file", type=str, required=True, help="Path to a text file containing the list of reference images to use during training and inference.")

# system args
parser.add_argument("--seed", type=int, default=42, help="Random seed. Default is 42.")
parser.add_argument("--outdir", type=str, required=True)

# model args
parser.add_argument("--arch", choices=["resnet18", "resnet34", "resnet50", "resnet101", "vit_b_16, vit_b_32"], default="resnet18")
parser.add_argument("--projector", action="store_true", help="Whether or not to train a projector")
parser.add_argument("--finetune", type=str, choices=["all", "linear", "proj"], default="all", help="Which parameters to finetune. Default is all.")
parser.add_argument("--eval_only", action="store_true", help="Whether the run is only for obtaining evaluations")
parser.add_argument("--pretrained", action="store_true", help="Use pretrained weights. Default is False.")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training. Default is 32.")
parser.add_argument('--weight_decay', type=float, default=0.001, help="L2 regularization")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate. Default is 1e-4.")
parser.add_argument("--epochs", type=int, default=10, help="Number of epochs. Default is 10.")
parser.add_argument("--early_stop", type=int, default=5, help="Early stopping patience. Default is 5.")
parser.add_argument("--resume", action="store_true", help="Whether to load a pretrained model from the give dir")
parser.add_argument("--checkpoint", type=str, help="Checkpoint path for resuming training, requires --resume to be passed as well")

def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def configure_output_directory(outdir, args):
    train_details = f"{args.dataset}/{args.comparison_type}_comparison/threshold_{args.threshold}_{args.arch}_batchsize_{args.batch_size}_{args.epochs}_epochs_lr_{args.lr}_seed_{args.seed}_weight_decay_{args.weight_decay}_fold_{args.fold}_data_prop_{args.data_prop}_task_{args.task}"
    if args.comparison_type == "offline":
        train_details += f"_{args.n_comparisons_per_image}_comparisons"
    if args.pretrained:
        train_details += "_pretrained"
    if args.projector:
        train_details += "_projector"
    train_details += f"_finetune_{args.finetune}" 
    out_dir = os.path.join(outdir, train_details)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


if __name__ == "__main__":
    args = parser.parse_args()

    if args.finetune == "proj" and not args.projector:
        raise ValueError("If --finetune is set to 'proj', --projector must also be set.")
    
    args.outdir = configure_output_directory(args.outdir, args)

    print(vars(args))
    logging.basicConfig(
        level=logging.INFO, filename=os.path.join(args.outdir, "log.txt")
    )

    setup_seed(args.seed)
    logging.info(f"Arguments: {vars(args)}")

    trainloader, inference_loaders_val, inference_loaders_test, ref_images, train_images, val_images, test_images = load_dataset(args)
    model = taco(args)
    model = model.cuda() if torch.cuda.is_available() else model

    if args.resume:
        model.load_state_dict(torch.load(args.checkpoint))
        logging.info(f"Resumed model from checkpoint: {args.checkpoint}")

    if not args.eval_only:
        trainlosses, trainaccs = model.train_model(trainloader, inference_loaders_val, ref_images, args.outdir)

        with open(os.path.join(args.outdir, "train_losses.txt"), "w") as f:
            for loss in trainlosses:
                f.write(f"{loss}\n")
        with open(os.path.join(args.outdir, "train_metrics.txt"), "w") as f:
            for acc in trainaccs:
                f.write(f"{acc}\n")

        # save the model
        torch.save(model.state_dict(), os.path.join(args.outdir, "final_model.pth"))
        logging.info("Model saved to %s", os.path.join(args.outdir, "final_model.pth"))

        model.load_state_dict(torch.load(os.path.join(args.outdir, "best_model.pth")))

    if "repr" not in args.task:
        metrics_ = {}
        logging.info(f"validating model")
        for loader, img in zip(inference_loaders_val, ref_images):
            logging.info(f"perf. (val set) - {img}")
            outfilename = args.outdir + f"/val_preds_{os.path.basename(img)}.csv"
            metrics_[img] = model.eval_model(loader, save=True, outfilename = outfilename, val_image_names = val_images)
        with open(os.path.join(args.outdir, "val_perf.json"), "w") as f:
            json.dump(metrics_, f)

        metrics_ = {}
        logging.info(f"validating model")
        for loader, img in zip(inference_loaders_test, ref_images):
            logging.info(f"perf. (test set) - {img}")
            outfilename = args.outdir + f"/test_preds_{os.path.basename(img)}.csv"
            metrics_[img] = model.eval_model(loader, save=True, outfilename = outfilename, val_image_names = test_images)
        with open(os.path.join(args.outdir, "test_perf.json"), "w") as f:
            json.dump(metrics_, f)
    else:
        logging.info("representation learning task; skipping evaluation.")

    logging.info("complete.")