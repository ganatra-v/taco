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
    if dataset in [
        "eyes-defy-anemia",
        "eyes-defy-anemia-india",
        "eyes-defy-anemia-italy",
    ]:
        subset = dataset.split("-")[-1] if dataset != "eyes-defy-anemia" else None
        trainloader, valloader = get_eyes_defy_anemia_dataloader(
            dataset_path,
            args,
            subset=subset,
        )
    elif dataset == "neojaundice":
        trainloader, valloader = get_neojaundice_dataloader(dataset_path)
    return trainloader, valloader


def get_eyes_defy_anemia_dataloader(dataset_path, args, subset=None):
    all_images = []
    all_labels = []

    if subset in [None, "india"]:
        india_images, india_labels = load_eyes_defy_anemia_country_data(
            "India", dataset_path
        )
        all_images.extend(india_images)
        all_labels.extend(india_labels)
    if subset in [None, "italy"]:
        italy_images, italy_labels = load_eyes_defy_anemia_country_data(
            "Italy", dataset_path
        )
        all_images.extend(italy_images)
        all_labels.extend(italy_labels)
    print(f"Total images: {len(all_images)}")
    print(
        f"Anemic = {sum([int(f<=args.anemia_threshold) for f in all_labels])}, Non-anemic = {len(all_labels) - sum([int(f<=args.anemia_threshold) for f in all_labels])}"
    )

    with open(os.path.join(args.outdir, "all_data.txt"), "w") as f:
        for img, label in zip(all_images, all_labels):
            f.write(f"{img},{label}\n")

    train_images, val_images, train_labels, val_labels = train_test_split(
        all_images,
        all_labels,
        test_size=0.2,
        random_state=42,
        stratify=[int(label <= args.anemia_threshold) for label in all_labels],
    )
    val_labels = [int(label <= args.anemia_threshold) for label in val_labels]

    images_1, images_2, comparison_labels = generate_comparisons(
        train_images, train_labels, n_comparisons_per_image=args.n_comparisons_per_image
    )
    comparison_data = pd.DataFrame(
        {"image_1": images_1, "image_2": images_2, "label": comparison_labels}
    )
    comparison_data.to_csv(
        os.path.join(args.outdir, "train_comparisons.csv"), index=False
    )

    comparison_dataset = ComparisonDataset(
        images_1, images_2, comparison_labels, split="train"
    )
    train_mean_, train_std = comparison_dataset.mean_, comparison_dataset.std_

    val_dataset = ComparisonDataset(
        val_images,
        val_images,
        val_labels,
        split="val",
        mean_=train_mean_,
        std_=train_std,
    )

    trainloader = DataLoader(
        comparison_dataset, batch_size=32, shuffle=True, num_workers=4
    )
    valloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    return trainloader, valloader


def load_eyes_defy_anemia_country_data(country, dataset_path):
    country_dir = os.path.join(dataset_path, country)
    images = []
    labels = []
    gt = pd.read_excel(os.path.join(country_dir, f"{country}.xlsx"))

    for row in tqdm(gt[["Number", "Hgb"]].iterrows()):
        img_dir = os.path.join(country_dir, str(int(row[1]["Number"])))
        label = row[1]["Hgb"]
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
            label = 1 if labels[images.index(img)] < labels[images.index(img2)] else 0
            comparison_labels.append(label)
    return images_1, images_2, comparison_labels


def get_neojaundice_dataloader(dataset_path):
    # Placeholder function for neojaundice dataset loading
    # Implement similar to get_eyes_defy_anemia_dataloader
    pass


class ComparisonDataset(Dataset):
    def __init__(self, images_1, images_2, labels, split, mean_=None, std_=None):
        print(f"Loading {split} dataset with {len(labels)} samples.")
        self.images_1 = [self.load_image(img) for img in images_1[:20]]
        self.images_2 = [self.load_image(img) for img in images_2[:20]]
        print("Completed loading images.")

        self.mean_ = (
            np.mean([np.mean(img, axis=(0, 1, 2)) for img in self.images_1])
            if mean_ is None
            else mean_
        )
        self.std_ = (
            np.mean([np.std(img, axis=(0, 1, 2)) for img in self.images_1])
            if std_ is None
            else std_
        )

        self.labels = labels[:20]
        self.split = split
        transforms_ = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[self.mean_ / 255.0], std=[self.std_ / 255.0]),
        ]
        if split == "train":
            transforms_.extend(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomRotation(20),
                    transforms.GaussianBlur(3),
                ]
            )
        self.transform = transforms.Compose(transforms_)

    def load_image(self, img_path):
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # resize image to 10% of original size
        image = cv2.resize(image, (0, 0), fx=0.1, fy=0.1)
        return image

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image_1 = self.images_1[idx]
        image_2 = self.images_2[idx] if self.split == "train" else self.images_1[idx]
        label = self.labels[idx]

        if self.transform:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)

        return image_1, image_2, torch.tensor(label, dtype=torch.float32)


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
