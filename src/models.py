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
            self.model.encoder.features[3],   # 24 channels
            self.model.encoder.features[6],   # 32 channels
            self.model.encoder.features[13],  # 96 channels
            self.model.encoder.features[17],  # 320 channels (FIXED: was [18])
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
    def __init__(self):
        super().__init__()

        student_channels = [24, 32, 96, 320]    # Correct!
        teacher_channels = [64, 128, 256, 512]  # Correct!

        self.adapters = nn.ModuleList([
            nn.Conv2d(s_ch, t_ch, kernel_size=1, bias=False)
            for s_ch, t_ch in zip(student_channels, teacher_channels)
        ])

    def forward(self, student_features):
        return [
            adapter(feat)
            for adapter, feat in zip(self.adapters, student_features)
        ]