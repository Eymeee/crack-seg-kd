import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
#  Blocs de base
# ─────────────────────────────────────────────

class ConvBNReLU(nn.Module):
    """Conv 3x3 → BatchNorm → ReLU"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    """MaxPool 2x2 → ConvBNReLU"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBNReLU(in_ch, out_ch),
        )

    def forward(self, x):
        return self.block(x)


class Up(nn.Module):
    """Upsample bilinéaire → concat → ConvBNReLU"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = ConvBNReLU(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ─────────────────────────────────────────────
#  Teacher : U-Net++
# ─────────────────────────────────────────────

class UNetPlusPlus(nn.Module):
    """
    U-Net++ (Teacher) — architecture dense avec connexions imbriquées.
    Les feature maps intermédiaires sont retournées pour la distillation.

    Filtres : [64, 128, 256, 512, 1024]
    """

    def __init__(self, in_ch: int = 3, out_ch: int = 1):
        super().__init__()

        f = [64, 128, 256, 512, 1024]

        # Encodeur
        self.enc0 = ConvBNReLU(in_ch, f[0])
        self.enc1 = Down(f[0], f[1])
        self.enc2 = Down(f[1], f[2])
        self.enc3 = Down(f[2], f[3])
        self.enc4 = Down(f[3], f[4])   # bottleneck

        # Nœuds denses (U-Net++ nested dense blocks)
        # Niveau 1
        self.x01 = Up(f[1] + f[0], f[0])
        # Niveau 2
        self.x11 = Up(f[2] + f[1], f[1])
        self.x02 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            ConvBNReLU(f[1] + f[0] * 2, f[0]),
        )
        # Niveau 3
        self.x21 = Up(f[3] + f[2], f[2])
        self.x12 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            ConvBNReLU(f[2] + f[1] * 2, f[1]),
        )
        self.x03 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            ConvBNReLU(f[1] + f[0] * 3, f[0]),
        )
        # Niveau 4
        self.x31 = Up(f[4] + f[3], f[3])
        self.x22 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            ConvBNReLU(f[3] + f[2] * 2, f[2]),
        )
        self.x13 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            ConvBNReLU(f[2] + f[1] * 2, f[1]),
        )
        self.x04 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            ConvBNReLU(f[1] + f[0] * 3, f[0]),
        )

        # Tête de segmentation
        self.head = nn.Conv2d(f[0], out_ch, kernel_size=1)

    def forward(self, x):
        # Encodeur
        x00 = self.enc0(x)
        x10 = self.enc1(x00)
        x20 = self.enc2(x10)
        x30 = self.enc3(x20)
        x40 = self.enc4(x30)

        # Décodeur dense
        _x01 = self.x01(x10, x00)
        _x11 = self.x11(x20, x10)
        _x02_in = torch.cat([
            F.interpolate(x10, size=_x01.shape[2:], mode="bilinear", align_corners=True),
            x00, _x01
        ], dim=1)
        _x02 = self.x02[1](self.x02[0](_x02_in))

        _x21 = self.x21(x30, x20)
        _x12_in = torch.cat([
            F.interpolate(x20, size=_x11.shape[2:], mode="bilinear", align_corners=True),
            x10, _x11
        ], dim=1)
        _x12 = self.x12[1](self.x12[0](_x12_in))
        _x03_in = torch.cat([
            F.interpolate(_x11, size=_x02.shape[2:], mode="bilinear", align_corners=True),
            x00, _x01, _x02
        ], dim=1)
        _x03 = self.x03[1](self.x03[0](_x03_in))

        _x31 = self.x31(x40, x30)
        _x22_in = torch.cat([
            F.interpolate(x30, size=_x21.shape[2:], mode="bilinear", align_corners=True),
            x20, _x21
        ], dim=1)
        _x22 = self.x22[1](self.x22[0](_x22_in))
        _x13_in = torch.cat([
            F.interpolate(_x21, size=_x12.shape[2:], mode="bilinear", align_corners=True),
            x10, _x11
        ], dim=1)
        _x13 = self.x13[1](self.x13[0](_x13_in))
        _x04_in = torch.cat([
            F.interpolate(_x12, size=_x03.shape[2:], mode="bilinear", align_corners=True),
            x00, _x01, _x02
        ], dim=1)
        _x04 = self.x04[1](self.x04[0](_x04_in))

        logits = self.head(_x04)

        # Feature maps retournées pour la distillation (4 niveaux encodeur)
        features = [x00, x10, x20, x30]

        return logits, features


# ─────────────────────────────────────────────
#  Student : U-Net léger
# ─────────────────────────────────────────────

class UNetStudent(nn.Module):
    """
    U-Net léger (Student) — architecture standard avec filtres réduits.
    Les feature maps intermédiaires sont retournées pour la distillation.

    Filtres : [32, 64, 128, 256]  (vs [64,128,256,512,1024] pour le Teacher)
    """

    def __init__(self, in_ch: int = 3, out_ch: int = 1):
        super().__init__()

        f = [32, 64, 128, 256]

        # Encodeur
        self.enc0 = ConvBNReLU(in_ch, f[0])
        self.enc1 = Down(f[0], f[1])
        self.enc2 = Down(f[1], f[2])
        self.enc3 = Down(f[2], f[3])   # bottleneck

        # Décodeur
        self.dec2 = Up(f[3] + f[2], f[2])
        self.dec1 = Up(f[2] + f[1], f[1])
        self.dec0 = Up(f[1] + f[0], f[0])

        # Tête de segmentation
        self.head = nn.Conv2d(f[0], out_ch, kernel_size=1)

    def forward(self, x):
        # Encodeur
        e0 = self.enc0(x)
        e1 = self.enc1(e0)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        # Décodeur
        d2 = self.dec2(e3, e2)
        d1 = self.dec1(d2, e1)
        d0 = self.dec0(d1, e0)

        logits = self.head(d0)

        # Feature maps retournées pour la distillation (4 niveaux encodeur)
        features = [e0, e1, e2, e3]

        return logits, features


# ─────────────────────────────────────────────
#  Couches d'adaptation (1×1 conv)
# ─────────────────────────────────────────────

class FeatureAdapters(nn.Module):
    """
    Projette les feature maps du Student vers les dimensions du Teacher
    via des convolutions 1×1, pour permettre le calcul de la KD loss.

    Canaux Teacher : [64, 128, 256, 512]
    Canaux Student : [32,  64, 128, 256]
    """

    def __init__(self):
        super().__init__()

        teacher_channels = [64, 128, 256, 512]
        student_channels = [32,  64, 128, 256]

        self.adapters = nn.ModuleList([
            nn.Conv2d(s_ch, t_ch, kernel_size=1, bias=False)
            for s_ch, t_ch in zip(student_channels, teacher_channels)
        ])

    def forward(self, student_features):
        """
        Args:
            student_features : liste de 4 tenseurs [e0, e1, e2, e3]
        Returns:
            liste de 4 tenseurs projetés dans l'espace du Teacher
        """
        return [
            adapter(feat)
            for adapter, feat in zip(self.adapters, student_features)
        ]


# ─────────────────────────────────────────────
#  Test rapide
# ─────────────────────────────────────────────

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(2, 3, 512, 512).to(device)

    # Teacher
    teacher = UNetPlusPlus().to(device)
    t_logits, t_feats = teacher(x)
    print("=== Teacher (U-Net++) ===")
    print(f"  Logits     : {t_logits.shape}")
    for i, f in enumerate(t_feats):
        print(f"  Feature {i}  : {f.shape}")

    # Student
    student = UNetStudent().to(device)
    s_logits, s_feats = student(x)
    print("\n=== Student (U-Net léger) ===")
    print(f"  Logits     : {s_logits.shape}")
    for i, f in enumerate(s_feats):
        print(f"  Feature {i}  : {f.shape}")

    # Adapters
    adapters = FeatureAdapters().to(device)
    adapted  = adapters(s_feats)
    print("\n=== Features adaptées (Student → espace Teacher) ===")
    for i, (a, t) in enumerate(zip(adapted, t_feats)):
        match = "✓" if a.shape == t.shape else "✗"
        print(f"  Niveau {i} : adapté {a.shape} | teacher {t.shape}  {match}")

    # Paramètres
    def count_params(m):
        return sum(p.numel() for p in m.parameters()) / 1e6

    print(f"\nTeacher  : {count_params(teacher):.1f} M paramètres")
    print(f"Student  : {count_params(student):.1f} M paramètres")
    print(f"Adapters : {count_params(adapters):.3f} M paramètres")