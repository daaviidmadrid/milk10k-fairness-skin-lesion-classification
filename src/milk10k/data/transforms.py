"""Image preprocessing and augmentation policies used in the experiments."""

from __future__ import annotations

from torchvision import transforms


# Build the deterministic preprocessing used for validation and test.
def eval_transform(image_size: int = 224) -> transforms.Compose:
    """Return the standard deterministic preprocessing pipeline."""
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ]
    )


# Build the baseline training transform without extra augmentation.
def train_transform(image_size: int = 224) -> transforms.Compose:
    """Return the default supervised training preprocessing."""
    return eval_transform(image_size)


# Build the online augmentation policy retained for the thesis experiments.
def online_augmentation_transform(image_size: int = 224) -> transforms.Compose:
    """Return a compact dermatology-compatible online augmentation policy."""
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomApply([transforms.RandomRotation(15)], p=0.6),
            transforms.RandomApply([transforms.RandomAffine(degrees=0, scale=(0.9, 1.1), shear=8)], p=0.5),
            transforms.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.06, hue=0.02),
            transforms.ToTensor(),
        ]
    )

