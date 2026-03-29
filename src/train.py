import os
import argparse
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset import get_dataloaders
from models  import UNetPlusPlus, UNetStudent, FeatureAdapters
from losses  import SegmentationLoss, TotalDistillationLoss
from metrics import evaluate, count_parameters, benchmark_inference, print_comparison_table


# ─────────────────────────────────────────────
#  Utilitaires
# ─────────────────────────────────────────────

def save_checkpoint(state, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    print(f"  ✓ Checkpoint sauvegardé → {path}")


def load_checkpoint(model, path, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    print(f"  ✓ Checkpoint chargé ← {path}  (epoch {ckpt['epoch']}, IoU {ckpt['iou']:.4f})")
    return model


# ─────────────────────────────────────────────
#  Mode 1 — Entraînement du Teacher
# ─────────────────────────────────────────────

def train_teacher(args, device, loaders):
    print("\n" + "="*55)
    print("  MODE : TEACHER (U-Net++)")
    print("="*55)

    model     = UNetPlusPlus().to(device)
    criterion = SegmentationLoss(alpha=0.5)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_iou  = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for images, masks in loaders["train"]:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            logits, _ = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()

        # Validation
        val_metrics = evaluate(model, loaders["val"], device)
        avg_loss    = total_loss / len(loaders["train"])

        print(f"  Epoch [{epoch:03d}/{args.epochs}]  "
              f"Loss: {avg_loss:.4f}  "
              f"IoU: {val_metrics['iou']:.4f}  "
              f"F1: {val_metrics['f1']:.4f}")

        # Sauvegarder le meilleur checkpoint
        if val_metrics["iou"] > best_iou:
            best_iou = val_metrics["iou"]
            save_checkpoint({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "iou":         best_iou,
            }, path="checkpoints/teacher_best.pth")

    print(f"\n  Meilleur IoU Teacher : {best_iou:.4f}")
    return model


# ─────────────────────────────────────────────
#  Mode 2 — Entraînement du Student seul
# ─────────────────────────────────────────────

def train_student_alone(args, device, loaders):
    print("\n" + "="*55)
    print("  MODE : STUDENT SEUL (sans distillation)")
    print("="*55)

    model     = UNetStudent().to(device)
    criterion = SegmentationLoss(alpha=0.5)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_iou  = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for images, masks in loaders["train"]:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            logits, _ = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()

        val_metrics = evaluate(model, loaders["val"], device)
        avg_loss    = total_loss / len(loaders["train"])

        print(f"  Epoch [{epoch:03d}/{args.epochs}]  "
              f"Loss: {avg_loss:.4f}  "
              f"IoU: {val_metrics['iou']:.4f}  "
              f"F1: {val_metrics['f1']:.4f}")

        if val_metrics["iou"] > best_iou:
            best_iou = val_metrics["iou"]
            save_checkpoint({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "iou":         best_iou,
            }, path="checkpoints/student_alone_best.pth")

    print(f"\n  Meilleur IoU Student seul : {best_iou:.4f}")
    return model


# ─────────────────────────────────────────────
#  Mode 3 — Distillation Feature-Based
# ─────────────────────────────────────────────

def train_distill(args, device, loaders):
    print("\n" + "="*55)
    print("  MODE : DISTILLATION (Feature-Based KD)")
    print("="*55)

    # Charger le Teacher pré-entraîné
    teacher = UNetPlusPlus().to(device)
    if not os.path.exists("checkpoints/teacher_best.pth"):
        raise FileNotFoundError(
            "Le checkpoint du Teacher est introuvable.\n"
            "Lancez d'abord : python train.py --mode teacher"
        )
    load_checkpoint(teacher, "checkpoints/teacher_best.pth", device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False          # Teacher figé

    # Student + Adapters
    student  = UNetStudent().to(device)
    adapters = FeatureAdapters().to(device)

    criterion = TotalDistillationLoss(lambda_kd=args.lambda_kd, alpha_seg=0.5)
    optimizer = optim.Adam(
        list(student.parameters()) + list(adapters.parameters()),
        lr=args.lr,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_iou  = 0.0

    for epoch in range(1, args.epochs + 1):
        student.train()
        adapters.train()

        total_loss = 0.0
        total_seg  = 0.0
        total_kd   = 0.0

        for images, masks in loaders["train"]:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()

            # Forward Teacher (sans gradient)
            with torch.no_grad():
                _, teacher_features = teacher(images)

            # Forward Student
            student_logits, student_features = student(images)

            # Adapter les features du Student
            adapted_features = adapters(student_features)

            # Calculer la loss totale
            loss, l_seg, l_kd = criterion(
                student_logits, masks,
                adapted_features, teacher_features,
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_seg  += l_seg.item()
            total_kd   += l_kd.item()

        scheduler.step()

        n          = len(loaders["train"])
        val_metrics = evaluate(student, loaders["val"], device)

        print(f"  Epoch [{epoch:03d}/{args.epochs}]  "
              f"Loss: {total_loss/n:.4f}  "
              f"(seg={total_seg/n:.4f}, kd={total_kd/n:.4f})  "
              f"IoU: {val_metrics['iou']:.4f}  "
              f"F1: {val_metrics['f1']:.4f}")

        if val_metrics["iou"] > best_iou:
            best_iou = val_metrics["iou"]
            save_checkpoint({
                "epoch":           epoch,
                "model_state":     student.state_dict(),
                "adapters_state":  adapters.state_dict(),
                "iou":             best_iou,
            }, path="checkpoints/student_distilled_best.pth")

    print(f"\n  Meilleur IoU Student distillé : {best_iou:.4f}")
    return student


# ─────────────────────────────────────────────
#  Mode 4 — Évaluation finale comparative
# ─────────────────────────────────────────────

def run_evaluation(args, device, loaders):
    print("\n" + "="*55)
    print("  MODE : ÉVALUATION COMPARATIVE")
    print("="*55)

    results = {}

    checkpoints = {
        "Teacher (U-Net++)"  : ("checkpoints/teacher_best.pth",           UNetPlusPlus),
        "Student seul"       : ("checkpoints/student_alone_best.pth",     UNetStudent),
        "Student distillé"   : ("checkpoints/student_distilled_best.pth", UNetStudent),
    }

    for name, (ckpt_path, ModelClass) in checkpoints.items():
        if not os.path.exists(ckpt_path):
            print(f"  ⚠  Checkpoint manquant pour {name} — ignoré")
            continue

        model = ModelClass().to(device)
        load_checkpoint(model, ckpt_path, device)

        metrics = evaluate(model, loaders["test"], device)
        latency = benchmark_inference(model, device, img_size=args.img_size)
        params  = count_parameters(model)

        results[name] = {
            "iou":        metrics["iou"],
            "f1":         metrics["f1"],
            "params":     params,
            "latency_ms": latency,
        }

    print_comparison_table(results)


# ─────────────────────────────────────────────
#  Point d'entrée
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Entraînement KD — Segmentation de fissures"
    )
    parser.add_argument(
        "--mode",
        choices=["teacher", "student", "distill", "eval"],
        required=True,
        help=(
            "teacher  : entraîner le Teacher U-Net++\n"
            "student  : entraîner le Student seul\n"
            "distill  : entraîner le Student avec Feature-Based KD\n"
            "eval     : évaluation comparative des 3 modèles"
        ),
    )
    parser.add_argument("--data_dir",   type=str,   default="data")
    parser.add_argument("--epochs",     type=int,   default=50)
    parser.add_argument("--batch_size", type=int,   default=8)
    parser.add_argument("--img_size",   type=int,   default=512)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--lambda_kd",  type=float, default=1.0,
                        help="Poids de la KD loss (mode distill uniquement)")
    parser.add_argument("--num_workers",type=int,   default=2)
    return parser.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device : {device}")

    loaders = get_dataloaders(
        data_dir=args.data_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    if args.mode == "teacher":
        train_teacher(args, device, loaders)

    elif args.mode == "student":
        train_student_alone(args, device, loaders)

    elif args.mode == "distill":
        train_distill(args, device, loaders)

    elif args.mode == "eval":
        run_evaluation(args, device, loaders)


if __name__ == "__main__":
    main()