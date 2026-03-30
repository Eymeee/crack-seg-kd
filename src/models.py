import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


# ─────────────────────────────────────────────
#  Teacher : U-Net++ (via segmentation-models-pytorch)
# ─────────────────────────────────────────────

class UNetPlusPlus(nn.Module):
    """
    Teacher — U-Net++ avec encodeur ResNet34 pré-entraîné.
    Retourne (logits, features) pour la distillation.
    """

    def __init__(self, in_ch: int = 3, out_ch: int = 1):
        super().__init__()

        self.model = smp.UnetPlusPlus(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=in_ch,
            classes=out_ch,
        )

        # Hooks pour extraire les feature maps intermédiaires
        self._features = []
        self._hooks    = []
        self._register_hooks()

    def _register_hooks(self):
        # Extraire les 4 premiers blocs de l'encodeur
        layers = [
            self.model.encoder.layer1,
            self.model.encoder.layer2,
            self.model.encoder.layer3,
            self.model.encoder.layer4,
        ]
        for layer in layers:
            h = layer.register_forward_hook(self._hook_fn)
            self._hooks.append(h)

    def _hook_fn(self, module, input, output):
        self._features.append(output)

    def forward(self, x):
        self._features = []
        logits = self.model(x)
        features = self._features.copy()
        return logits, features


# ─────────────────────────────────────────────
#  Student : U-Net léger (via segmentation-models-pytorch)
# ─────────────────────────────────────────────

class UNetStudent(nn.Module):
    """
    Student — U-Net léger avec encodeur MobileNetV2 pré-entraîné.
    Retourne (logits, features) pour la distillation.
    """

    def __init__(self, in_ch: int = 3, out_ch: int = 1):
        super().__init__()

        self.model = smp.Unet(
            encoder_name="mobilenet_v2",
            encoder_weights="imagenet",
            in_channels=in_ch,
            classes=out_ch,
        )

        self._features = []
        self._hooks    = []
        self._register_hooks()

    def _register_hooks(self):
        # Extraire 4 blocs de l'encodeur MobileNetV2
        layers = [
            self.model.encoder.features[3],
            self.model.encoder.features[6],
            self.model.encoder.features[13],
            self.model.encoder.features[18],
        ]
        for layer in layers:
            h = layer.register_forward_hook(self._hook_fn)
            self._hooks.append(h)

    def _hook_fn(self, module, input, output):
        self._features.append(output)

    def forward(self, x):
        self._features = []
        logits = self.model(x)
        features = self._features.copy()
        return logits, features


# ─────────────────────────────────────────────
#  Couches d'adaptation (1×1 conv)
# ─────────────────────────────────────────────

class FeatureAdapters(nn.Module):
    """
    Projette les feature maps du Student vers les dimensions du Teacher
    via des convolutions 1×1.

    Teacher (ResNet34)    : [64, 128, 256, 512]
    Student (MobileNetV2) : [24,  32,  96, 320]
    """

    def __init__(self):
        super().__init__()

        student_channels = [24,  32,  96, 320]
        teacher_channels = [64, 128, 256, 512]

        self.adapters = nn.ModuleList([
            nn.Conv2d(s_ch, t_ch, kernel_size=1, bias=False)
            for s_ch, t_ch in zip(student_channels, teacher_channels)
        ])

    def forward(self, student_features):
        return [
            adapter(feat)
            for adapter, feat in zip(self.adapters, student_features)
        ]


# ─────────────────────────────────────────────
#  Test rapide
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(2, 3, 256, 256).to(device)

    teacher  = UNetPlusPlus().to(device)
    student  = UNetStudent().to(device)
    adapters = FeatureAdapters().to(device)

    with torch.no_grad():
        t_logits, t_feats = teacher(x)
        s_logits, s_feats = student(x)
        adapted           = adapters(s_feats)

    print("=== Teacher (U-Net++ / ResNet34) ===")
    print(f"  Logits : {t_logits.shape}")
    for i, f in enumerate(t_feats):
        print(f"  Feature {i} : {f.shape}")

    print("\n=== Student (U-Net / MobileNetV2) ===")
    print(f"  Logits : {s_logits.shape}")
    for i, f in enumerate(s_feats):
        print(f"  Feature {i} : {f.shape}")

    print("\n=== Adaptation Student → Teacher ===")
    for i, (a, t) in enumerate(zip(adapted, t_feats)):
        ch_match = "✓" if a.shape[1] == t.shape[1] else "✗"
        print(f"  Niveau {i} : {tuple(a.shape)} → {tuple(t.shape)}  {ch_match}")

    def count_params(m):
        return sum(p.numel() for p in m.parameters()) / 1e6

    print(f"\nTeacher  : {count_params(teacher):.1f} M paramètres")
    print(f"Student  : {count_params(student):.1f} M paramètres")
    print(f"Adapters : {count_params(adapters):.3f} M paramètres")