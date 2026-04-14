import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
#  Loss de segmentation : Dice + BCE
# ─────────────────────────────────────────────

class DiceLoss(nn.Module):
    """
    Dice Loss pour la segmentation binaire.
    Mesure le chevauchement entre la prédiction et le masque réel.
    """

    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        Args:
            logits  : sortie brute du modèle [B, 1, H, W]
            targets : masques binaires        [B, 1, H, W]
        Returns:
            scalar — Dice Loss ∈ [0, 1]
        """
        probs = torch.sigmoid(logits)

        probs   = probs.view(-1)
        targets = targets.view(-1)

        intersection = (probs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / \
               (probs.sum() + targets.sum() + self.smooth)

        return 1.0 - dice


class SegmentationLoss(nn.Module):
    """
    Loss de segmentation combinée : Dice + BCE.

    L_seg = α * DiceLoss + (1 - α) * BCEWithLogitsLoss
    """

    def __init__(self, alpha: float = 0.5):
        """
        Args:
            alpha : poids du Dice Loss (1 - alpha pour la BCE)
        """
        super().__init__()
        self.alpha    = alpha
        self.dice     = DiceLoss()
        self.bce      = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        """
        Args:
            logits  : sortie brute du modèle [B, 1, H, W]
            targets : masques binaires        [B, 1, H, W]
        Returns:
            scalar — loss combinée
        """
        return self.alpha * self.dice(logits, targets) + \
               (1 - self.alpha) * self.bce(logits, targets)


# ─────────────────────────────────────────────
#  Loss de distillation : Feature-Based KD
# ─────────────────────────────────────────────

class FeatureKDLoss(nn.Module):
    """
    Feature-Based Knowledge Distillation Loss.

    Pour chaque niveau i, calcule le MSE entre la feature map adaptée
    du Student et la feature map du Teacher (détachée du graphe).

    L_kd = (1/N) * Σ_i MSE(Adapter(F_student_i), F_teacher_i)
    """

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, student_adapted_features, teacher_features):
        """
        Args:
            student_adapted_features : liste de N tenseurs (sortie des Adapters)
            teacher_features         : liste de N tenseurs (feature maps Teacher)
        Returns:
            scalar — moyenne des MSE sur tous les niveaux
        """
        assert len(student_adapted_features) == len(teacher_features), \
            "Le nombre de niveaux Student et Teacher doit être identique"

        total_loss = 0.0
        for s_feat, t_feat in zip(student_adapted_features, teacher_features):
            # Détacher le Teacher : il ne doit pas recevoir de gradient
            t_feat_detached = t_feat.detach()

            # Aligner les résolutions si nécessaire (sécurité)
            if s_feat.shape[2:] != t_feat_detached.shape[2:]:
                s_feat = F.interpolate(
                    s_feat,
                    size=t_feat_detached.shape[2:],
                    mode="bilinear",
                    align_corners=True,
                )

            total_loss += self.mse(s_feat, t_feat_detached)

        return total_loss / len(teacher_features)


# ─────────────────────────────────────────────
#  Loss totale : Segmentation + KD
# ─────────────────────────────────────────────

class TotalDistillationLoss(nn.Module):
    """
    Loss totale pour l'entraînement du Student avec distillation.

    L_total = L_seg + λ * L_kd

    où :
        L_seg = Dice + BCE  (sur les prédictions du Student vs vrai masque)
        L_kd  = MSE moyen   (feature maps Student adaptées vs Teacher)
        λ     = poids de la distillation (hyperparamètre)
    """

    def __init__(self, lambda_kd: float = 1.0, alpha_seg: float = 0.5):
        """
        Args:
            lambda_kd : poids de la KD loss
            alpha_seg : poids du Dice dans la SegmentationLoss
        """
        super().__init__()
        self.lambda_kd  = lambda_kd
        self.seg_loss   = SegmentationLoss(alpha=alpha_seg)
        self.kd_loss    = FeatureKDLoss()

    def forward(self,
                student_logits,
                targets,
                student_adapted_features,
                teacher_features):
        """
        Args:
            student_logits           : logits Student          [B, 1, H, W]
            targets                  : masques réels           [B, 1, H, W]
            student_adapted_features : features adaptées Student (liste)
            teacher_features         : features Teacher         (liste)
        Returns:
            total  : loss totale (scalar)
            l_seg  : composante segmentation (pour le logging)
            l_kd   : composante distillation (pour le logging)
        """
        l_seg = self.seg_loss(student_logits, targets)
        l_kd  = self.kd_loss(student_adapted_features, teacher_features)

        total = l_seg + self.lambda_kd * l_kd

        return total, l_seg, l_kd

