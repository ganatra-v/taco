import cv2
import os
import pandas as pd
import random
import logging
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


def load_dataset(args):
    return fetch_dataloaders(args)

def label_fn(args, x1, x2):
    if args.dataset == "eyes-defy-anemia":
        return x1<=x2 if "cls" in args.task else x2-x1
    elif args.dataset == "neojaundice":
        return x1>=x2 if "cls" in args.task else x1 - x2  
def fetch_dataloaders(args):
    train_data, val_data, test_data = get_split_data(args)
    logging.info(f"Trainset = {len(train_data)}, Valset = {len(val_data)} Testset = {len(test_data)}")
    logging.info(f"Train = {train_data["label"].value_counts()}, Val = {val_data["label"].value_counts()} Test = {test_data["label"].value_counts()}")

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
    label_key = "hgb" if args.dataset == "eyes-defy-anemia" else "tsb"
    train_labels = train_data[label_key].values


    # in both datasets, for classification, label is 1 if first image is diseased, i.e. has lower Hgb or higher TsB
    # in both datasets, for regression, label values are positive if first image is diseased, i.e. has lower Hgb or higher TsB
    trainset = ComparisonDataset(train_images, train_labels, args, label_fn)

    ref_images = []
    with open(args.ref_image_file, "r") as f:
        for line in f:
            ref_images.append(line.strip())
    assert len(ref_images) > 0, "No reference images found."

    val_images = val_data["images"].apply(lambda x: os.path.join(args.dataset_path, x)).values if args.dataset == "neojaundice" else val_data["images"].values
    val_labels = val_data[label_key].values

    test_images = test_data["images"].apply(lambda x: os.path.join(args.dataset_path, x)).values if args.dataset == "neojaundice" else test_data["images"].values
    test_labels = test_data[label_key].values

    logging.info(f"loading inference datasets for {len(ref_images)} ref images")    

    inference_loaders_val = []
    inference_loaders_test = []

    for img in ref_images:
        logging.info(f"ref image  - {img}")
        inference_dataset = InferenceDataset(args, val_images, img, val_labels, args.threshold, label_fn)
        trainloader = DataLoader(inference_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
        inference_loaders_val.append(trainloader)

        inference_dataset = InferenceDataset(args, test_images, img, test_labels, args.threshold, label_fn)
        testloader = DataLoader(inference_dataset, batch_size = 10, shuffle=False, num_workers=4)
        inference_loaders_test.append(testloader)
    
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    return trainloader, inference_loaders_val, inference_loaders_test, ref_images, train_images, val_images, test_images


def get_split_data(args):
    train_file = os.path.join(args.fold_path, f"train_fold_{args.fold}.csv")
    if args.dataset == "eyes-defy-anemia":
        val_file = train_file  # No separate validation file for this dataset
    else:
        val_file = os.path.join(args.fold_path, f"val_fold_{args.fold}.csv")
    test_file = os.path.join(args.fold_path, f"test_fold_{args.fold}.csv")

    train_data = pd.read_csv(train_file)
    val_data = pd.read_csv(val_file)
    test_data = pd.read_csv(test_file)
    return train_data, val_data, test_data

def generate_offline_comparisons(args, images, labels, n_comparisons_per_image, compare_fn):
    images_1 = []
    images_2 = []
    comparison_labels = []
    n = len(images)
    logging.info(
        f"Generating {n_comparisons_per_image} comparisons per image for {n} images."
    )
    for img in tqdm(images):
        other_images = [im for im in images if im != img]
        for _ in range(n_comparisons_per_image):
            img2 = random.choice(other_images)
            images_1.append(img)
            images_2.append(img2)
            # i.e. label is 1 if the first image has lower Hgb than the second image.
            # label is 1 if the first image has a higher TsB than second image
            label = compare_fn(args, labels[images.index(img)], labels[images.index(img2)])
            comparison_labels.append(label)
    return images_1, images_2, comparison_labels


class InferenceDataset(Dataset):
    def __init__(self, args, images, ref_image, labels, threshold, label_fn, mean_= [0.485, 0.456, 0.406], std_= [0.229, 0.224, 0.225]):
        logging.info(f"Loading dataset for inference using comparison")
        self.args = args
        self.images = [load_image(img) for img in tqdm(images)]
        self.ref_image = load_image(ref_image)
        self.threshold = threshold
        self.label_fn = label_fn
        logging.info("Completed loading images")
        self.mean_ = mean_
        self.std_ = std_

        self.labels = labels
        transforms_ = [
            transforms.ToTensor(),
            transforms.Normalize(mean= self.mean_ , std=self.std_),
        ]
        self.transform = transforms.Compose(transforms_)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image_1 = self.images[idx]
        image_2 = self.ref_image
        label = self.label_fn(self.args, self.labels[idx], self.threshold)

        if self.transform:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)

        return image_1, image_2, torch.tensor(label, dtype=torch.float32)

class ComparisonDataset(Dataset):
    # labels in this dataset correspond to continuous values (Hgb or TsB)
    def __init__(self, images, label_vals, args, label_fn, mean_= [0.485, 0.456, 0.406], std_= [0.229, 0.224, 0.225]):
        assert len(images) == len(label_vals), "Number of images and labels must be the same."
        logging.info(f"Loading dataset with {len(label_vals)} samples.")
        self.images = [load_image(img) for img in tqdm(images)]
        self.labels = label_vals
        self.comparison_type = args.comparison_type
        self.label_fn = label_fn
        self.args = args
        
        self.images_1 = range(len(images))
        if args.comparison_type == "offline":
            # Load pre-generated comparisons
            images_idx = range(len(images))            
            images_1, images_2, comparison_labels = generate_offline_comparisons(args, images_idx, label_vals, args.n_comparisons_per_image, self.label_fn)
            comparison_data = pd.DataFrame({
                "image_1": images[images_1], 
                "image_2": images[images_2], 
                "label": comparison_labels
            })
            logging.info(f"comparison labels generated: {comparison_data["label"].value_counts()}")
            comparison_data.to_csv(os.path.join(args.outdir, "train_comparisons.csv"), index=False)        
            self.images_1 = images_1
            self.images_2 = images_2
            self.labels = comparison_labels

        transforms_ = [
            transforms.ToTensor(),
            transforms.Normalize(mean = mean_ , std = std_),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
        ]

        self.transform = transforms.Compose(transforms_)     
        logging.info(f"{len(self.images)} images, {args.n_comparisons_per_image} comparisons per image, total {len(self.labels)}")       
        logging.info("Completed loading images.")


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image_1 = self.images[self.images_1[idx]]
        if self.comparison_type == "online":
            other_indices = [i for i in range(len(self.images)) if i != self.images_1[idx]]
            image_2_idx = random.choice(other_indices)
            image_2 = self.images[image_2_idx]
            label = self.label_fn(self.args, self.labels[self.images_1[idx]], self.labels[image_2_idx])
        else:
            image_2 = self.images[self.images_2[idx]]
            label = self.labels[idx]

        image_1 = self.transform(image_1)
        image_2 = self.transform(image_2)

        return image_1, image_2, torch.tensor(label, dtype=torch.float32)


def load_image(img_path):
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (256, 256))
    return image


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset")
    parser.add_argument(
        "--dataset_path", required=True, help="Path to the dataset you want to use."
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=10.5,
        help="Threshold for anemia in g/dL. Default is 11 g/dL.",
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
        help="Output directory to save comparison CSV files.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training. Default is 32.",
    )
    args = parser.parse_args()
    load_dataset(args)
