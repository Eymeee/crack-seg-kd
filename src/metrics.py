import torch
import numpy as np
from tabulate import tabulate


# ─────────────────────────────────────────────
#  Métriques de base
# ─────────────────────────────────────────────

def iou_score(logits, targets, threshold: float = 0.5, smooth: float = 1e-6):
    """
    Intersection over Union (IoU) — aussi appelé Jaccard Index.

    Args:
        logits    : sortie brute du modèle [B, 1, H, W]
        targets   : masques binaires       [B, 1, H, W]
        threshold : seuil de binarisation des prédictions
        smooth    : terme de stabilité numérique
    Returns:
        scalar — IoU moyen sur le batch
    """
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    preds   = preds.view(-1)
    targets = targets.view(-1)

    intersection = (preds * targets).sum()
    union        = preds.sum() + targets.sum() - intersection

    return ((intersection + smooth) / (union + smooth)).item()


def f1_score(logits, targets, threshold: float = 0.5, smooth: float = 1e-6):
    """
    F1-Score (Dice Coefficient) pour la segmentation binaire.

    Args:
        logits    : sortie brute du modèle [B, 1, H, W]
        targets   : masques binaires       [B, 1, H, W]
        threshold : seuil de binarisation des prédictions
        smooth    : terme de stabilité numérique
    Returns:
        scalar — F1 moyen sur le batch
    """
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    preds   = preds.view(-1)
    targets = targets.view(-1)

    tp = (preds * targets).sum()
    fp = (preds * (1 - targets)).sum()
    fn = ((1 - preds) * targets).sum()

    precision = (tp + smooth) / (tp + fp + smooth)
    recall    = (tp + smooth) / (tp + fn + smooth)

    return (2 * precision * recall / (precision + recall + smooth)).item()


# ─────────────────────────────────────────────
#  Évaluation sur un DataLoader complet
# ─────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, dataloader, device, threshold: float = 0.5):
    """
    Évalue un modèle sur un DataLoader complet.

    Args:
        model      : modèle PyTorch (Teacher ou Student)
        dataloader : DataLoader du split à évaluer
        device     : torch.device
        threshold  : seuil de binarisation
    Returns:
        dict {"iou": float, "f1": float}
    """
    model.eval()

    iou_list = []
    f1_list  = []

    for images, masks in dataloader:
        images = images.to(device)
        masks  = masks.to(device)

        outputs = model(images)

        # Gérer le cas où le modèle retourne (logits, features)
        if isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs

        iou_list.append(iou_score(logits, masks, threshold))
        f1_list.append(f1_score(logits, masks, threshold))

    return {
        "iou": float(np.mean(iou_list)),
        "f1":  float(np.mean(f1_list)),
    }


# ─────────────────────────────────────────────
#  Tableau comparatif final
# ─────────────────────────────────────────────

def count_parameters(model):
    """Retourne le nombre de paramètres en millions."""
    return sum(p.numel() for p in model.parameters()) / 1e6


@torch.no_grad()
def benchmark_inference(model, device, img_size: int = 512,
                         batch_size: int = 1, n_runs: int = 50):
    """
    Mesure la latence d'inférence moyenne en millisecondes.

    Args:
        model      : modèle PyTorch
        device     : torch.device
        img_size   : taille de l'image en entrée (carré)
        batch_size : taille du batch
        n_runs     : nombre de passes pour moyenner
    Returns:
        float — latence moyenne en ms
    """
    import time
    model.eval()
    dummy = torch.randn(batch_size, 3, img_size, img_size).to(device)

    # Warm-up
    for _ in range(5):
        _ = model(dummy)

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        _ = model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)

    return float(np.mean(times))


def print_comparison_table(results: dict):
    """
    Affiche un tableau comparatif des 3 modèles.

    Args:
        results : dict de la forme
            {
              "Teacher"          : {"iou": 0.82, "f1": 0.86, "params": 31.2, "latency_ms": 45.1},
              "Student seul"     : {"iou": 0.74, "f1": 0.79, "params":  3.1, "latency_ms": 12.3},
              "Student distillé" : {"iou": 0.80, "f1": 0.84, "params":  3.1, "latency_ms": 12.3},
            }
    """
    rows = []
    for model_name, metrics in results.items():
        rows.append([
            model_name,
            f"{metrics['iou']:.4f}",
            f"{metrics['f1']:.4f}",
            f"{metrics['params']:.1f} M",
            f"{metrics['latency_ms']:.1f} ms",
        ])

    headers = ["Modèle", "IoU ↑", "F1 ↑", "Paramètres ↓", "Latence ↓"]
    print("\n" + tabulate(rows, headers=headers, tablefmt="rounded_outline"))

