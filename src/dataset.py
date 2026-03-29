import os
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader


# ─────────────────────────────────────────────
#  Augmentations
# ─────────────────────────────────────────────

def get_transforms(split: str, img_size: int = 512):
    """
    Retourne les augmentations selon le split :
    - train : augmentations aléatoires
    - val / test : redimensionnement + normalisation uniquement
    """
    if split == "train":
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.GaussianBlur(p=0.2),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])


# ─────────────────────────────────────────────
#  Dataset
# ─────────────────────────────────────────────

class CrackDataset(Dataset):
    """
    Dataset de segmentation de fissures.

    Structure attendue dans data/ :
        data/{split}/images/  ← images RGB (.jpg ou .png)
        data/{split}/masks/   ← masques binaires (.jpg ou .png)

    Les masques sont binarisés : pixel > 127 → 1 (fissure), sinon 0.
    """

    def __init__(self, data_dir: str, split: str, img_size: int = 512):
        """
        Args:
            data_dir  : chemin vers le dossier data/  (ex: "data")
            split     : "train", "val" ou "test"
            img_size  : taille cible des images (carré)
        """
        assert split in ("train", "val", "test"), \
            f"split doit être 'train', 'val' ou 'test', reçu : {split}"

        self.images_dir = os.path.join(data_dir, split, "images")
        self.masks_dir  = os.path.join(data_dir, split, "masks")
        self.transform  = get_transforms(split, img_size)

        # Lister et trier les fichiers pour assurer la correspondance image/masque
        self.image_files = sorted([
            f for f in os.listdir(self.images_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

        assert len(self.image_files) > 0, \
            f"Aucune image trouvée dans {self.images_dir}"

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        fname = self.image_files[idx]

        # Charger l'image en RGB
        image_path = os.path.join(self.images_dir, fname)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Charger le masque en niveaux de gris
        mask_path = os.path.join(self.masks_dir, fname)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Binariser le masque : fissure = 1, fond = 0
        mask = (mask > 127).astype(np.float32)

        # Appliquer les augmentations
        augmented = self.transform(image=image, mask=mask)
        image = augmented["image"]          # Tensor [3, H, W]
        mask  = augmented["mask"].unsqueeze(0)  # Tensor [1, H, W]

        return image, mask


# ─────────────────────────────────────────────
#  DataLoaders
# ─────────────────────────────────────────────

def get_dataloaders(data_dir: str,
                    img_size: int = 512,
                    batch_size: int = 8,
                    num_workers: int = 2):
    """
    Retourne les DataLoaders train, val et test.

    Args:
        data_dir    : chemin vers le dossier data/
        img_size    : taille des images en entrée
        batch_size  : taille des batchs
        num_workers : workers pour le chargement parallèle

    Returns:
        dict avec les clés "train", "val", "test"
    """
    loaders = {}
    for split in ("train", "val", "test"):
        dataset = CrackDataset(data_dir, split, img_size)
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=True,
        )
        print(f"[{split}] {len(dataset)} images chargées")

    return loaders


# ─────────────────────────────────────────────
#  Test rapide
# ─────────────────────────────────────────────

if __name__ == "__main__":
    loaders = get_dataloaders(data_dir="data", img_size=512, batch_size=4)
    images, masks = next(iter(loaders["train"]))
    print(f"Batch images : {images.shape}")   # [4, 3, 512, 512]
    print(f"Batch masques : {masks.shape}")   # [4, 1, 512, 512]
    print(f"Valeurs masque : {masks.unique()}")  # tensor([0., 1.])