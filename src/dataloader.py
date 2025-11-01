import cv2
import os
import pandas as pd
import random
import logging
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


def load_dataset(args):
    dataset = args.dataset
    dataset_path = args.dataset_path
    if dataset =="eyes-defy-anemia":
        trainloader, infer_trainloaders, infer_val_loaders, reference_images, val_images = get_eyes_defy_anemia_dataloader(
            dataset_path,
            args
        )
    elif dataset == "neojaundice":
        trainloader, infer_trainloaders, infer_val_loaders, reference_images, val_images = get_neojaundice_dataloader(dataset_path)
    return trainloader, infer_trainloaders, infer_val_loaders, reference_images, val_images


def get_eyes_defy_anemia_dataloader(dataset_path, args):
    train_file = os.path.join(args.fold_path, f"train_fold_{args.fold}.csv")
    test_file = os.path.join(args.fold_path, f"test_fold_{args.fold}.csv")

    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)

    logging.info(f"Trainset = {len(train_data)}, Testset = {len(test_data)}")
    logging.info(f"Train = {train_data["label"].value_counts()}, Test = {test_data["label"].value_counts()}")

    train_images = train_data["images"].values
    train_labels = train_data["label"].values
    train_hgb = train_data["hgb"].values

    test_images = test_data["images"].values
    test_labels = test_data["label"].values
    test_hgb = test_data["hgb"].values

    reference_images = [
        train_images[i]
        for i in range(len(train_images))
        if train_hgb[i] == args.anemia_threshold
    ]

    train_images = [load_image(img) for img in tqdm(train_images)]
    test_images = [load_image(img) for img in tqdm(test_images)]
    train_images_idx = range(len(train_images))

    images_1, images_2, comparison_labels = generate_comparisons(
        train_images_idx, train_hgb, n_comparisons_per_image=args.n_comparisons_per_image
    )
    comparison_data = pd.DataFrame(
        {"image_1": images_1, "image_2": images_2, "label": comparison_labels}
    )
    logging.info(f"comparison labels generated: {comparison_data["label"].value_counts()}")
    comparison_data.to_csv(
        os.path.join(args.outdir, "train_comparisons.csv"), index=False
    )

    # train_mean_, train_std = comparison_dataset.mean_, comparison_dataset.std_
    train_mean_ =[0.485, 0.456, 0.406]
    train_std_ = [0.229, 0.224, 0.225]

    logging.info("loading comparison datasets")
    comparison_dataset = ComparisonDataset(
        train_images, images_1, images_2, comparison_labels, mean_ = train_mean_, std_ = train_std_ 
    )


    logging.info(f"loading inference datasets for {len(reference_images)} ref images")    
    inference_loaders_train = []
    inference_loaders_val = []
    for img in reference_images:
        logging.info(f"ref image  - {img}")
        inference_dataset = InferenceDataset(train_images, img, train_labels, train_mean_, train_std_)
        trainloader = DataLoader(
            inference_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
        )
        inference_loaders_train.append(trainloader)

        inference_dataset = InferenceDataset(test_images, img, test_labels, train_mean_, train_std_)
        valloader = DataLoader(inference_dataset, batch_size = 10, shuffle=False, num_workers=4)
        inference_loaders_val.append(valloader)
 
    trainloader = DataLoader(
        comparison_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
    )

    return trainloader, inference_loaders_train, inference_loaders_val, reference_images, test_images


def load_eyes_defy_anemia_country_data(country, dataset_path):
    country_dir = os.path.join(dataset_path, country)
    images = []
    labels = []
    gt = pd.read_excel(os.path.join(country_dir, f"{country}.xlsx"))

    for row in tqdm(gt[["Number", "Hgb"]].iterrows()):
        img_dir = os.path.join(country_dir, str(int(row[1]["Number"])))
        label = float(row[1]["Hgb"])
        imgs = os.listdir(img_dir)
        try:
            # check for only numerical image names
            image = [f for f in imgs if "al" not in f and not f.startswith(".")][0]
            image_path = os.path.join(img_dir, image)
            images.append(image_path)
            labels.append(label)

        except IndexError:
            continue

    return images, labels

def generate_comparisons(images, labels, n_comparisons_per_image):
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
            label = 1 if labels[images.index(img)] <= labels[images.index(img2)] else 0
            comparison_labels.append(label)
    return images_1, images_2, comparison_labels


def get_neojaundice_dataloader(dataset_path):
    # Placeholder function for neojaundice dataset loading
    # Implement similar to get_eyes_defy_anemia_dataloader
    pass

class InferenceDataset(Dataset):
    def __init__(self, images, ref_image, labels, mean_ = None, std_ = None):
        logging.info(f"Loading dataset for inference using comparison")
        self.images = images
        self.ref_image = load_image(ref_image)
        logging.info("Completed loading images")
        if mean_ is not None and std_ is not None:
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
        label = self.labels[idx]

        if self.transform:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)

        return image_1, image_2, torch.tensor(label, dtype=torch.float32)

class ComparisonDataset(Dataset):
    def __init__(self, all_images, images_1, images_2, labels, mean_=None, std_=None):
        logging.info(f"Loading dataset with {len(labels)} samples.")
        self.all_images = all_images
        self.images_1 = images_1
        self.images_2 = images_2
        logging.info("Completed loading images.")

        if mean_ is not None and std_ is not None:
            self.mean_ = mean_
            self.std_ = std_

        self.labels = labels
        transforms_ = [
            transforms.ToTensor(),
            transforms.Normalize(mean= self.mean_ , std=self.std_),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
        ]
        self.transform = transforms.Compose(transforms_)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image_1 = self.all_images[self.images_1[idx]]
        image_2 = self.all_images[self.images_2[idx]]
        label = self.labels[idx]

        if self.transform:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)

        return image_1, image_2, torch.tensor(label, dtype=torch.float32)


def load_image(img_path):
    img_path = img_path.replace("/scratch/thirty3/vaibhavg/taco","../../")
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
        "--anemia_threshold",
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
