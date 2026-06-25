import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.GroupNorm(min(8, channels), channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.GroupNorm(min(8, channels), channels),
        )
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, num_res: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1, bias=False)
        self.norm = nn.GroupNorm(min(8, out_ch), out_ch)
        self.act = nn.SiLU()
        self.res = nn.Sequential(*[ResBlock(out_ch) for _ in range(num_res)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.res(self.act(self.norm(self.conv(x))))


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, num_res: int = 2):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.SiLU(),
        )
        self.res = nn.Sequential(*[ResBlock(out_ch) for _ in range(num_res)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.res(self.up(x))


class Encoder(nn.Module):
    """
    256 → 128 → 64 → 32 → 16 → 8 → 4  (spatial), then flatten → latent vector.
    3 → base_ch → 2x → 4x → 8x → 8x → 8x → 8x  (channels)

    The two extra downsamples past 16×16 collapse the spatial grid to 4×4 so the
    bottleneck is a *flat* latent vector (AAE-style), not a spatial map.  A flat
    latent is what lets us split it into an AAE-regularised half and an
    ArcFace-supervised identity half.
    """

    def __init__(self, base_ch: int = 64, num_res: int = 2,
                 latent_dim: int = 1024, image_size: int = 256):
        super().__init__()
        C = base_ch
        self.feat = image_size // 64  # 256 → 4
        self.stem = nn.Sequential(
            nn.Conv2d(3, C, 3, 1, 1, bias=False),
            nn.GroupNorm(min(8, C), C),
            nn.SiLU(),
            *[ResBlock(C) for _ in range(num_res)],
        )
        self.downs = nn.ModuleList([
            DownBlock(C,     C * 2, num_res),   # 128
            DownBlock(C * 2, C * 4, num_res),   # 64
            DownBlock(C * 4, C * 8, num_res),   # 32
            DownBlock(C * 8, C * 8, num_res),   # 16
            DownBlock(C * 8, C * 8, num_res),   # 8
            DownBlock(C * 8, C * 8, num_res),   # 4
        ])
        self.to_latent = nn.Linear(C * 8 * self.feat * self.feat, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for down in self.downs:
            x = down(x)
        return self.to_latent(x.flatten(1))


class Decoder(nn.Module):
    """
    latent vector → (base_ch*8, 4, 4) → 8 → 16 → 32 → 64 → 128 → 256  (spatial)
    Mirror of the Encoder.
    """

    def __init__(self, base_ch: int = 64, num_res: int = 2,
                 latent_dim: int = 1024, image_size: int = 256):
        super().__init__()
        C = base_ch
        self.feat = image_size // 64  # 4
        self.C8   = C * 8
        self.from_latent = nn.Linear(latent_dim, self.C8 * self.feat * self.feat)
        self.ups = nn.ModuleList([
            UpBlock(C * 8, C * 8, num_res),   # 8
            UpBlock(C * 8, C * 8, num_res),   # 16
            UpBlock(C * 8, C * 8, num_res),   # 32
            UpBlock(C * 8, C * 4, num_res),   # 64
            UpBlock(C * 4, C * 2, num_res),   # 128
            UpBlock(C * 2, C,     num_res),   # 256
        ])
        self.head = nn.Sequential(
            nn.Conv2d(C, 3, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.from_latent(z).view(-1, self.C8, self.feat, self.feat)
        for up in self.ups:
            x = up(x)
        return self.head(x)


class FaceAutoencoder(nn.Module):
    """
    Adversarial autoencoder (AAE) with a split latent for generalising to
    *unseen* identities.

    The flat latent of size `latent_dim` is split into two halves:
      - z_style (first `style_dim` dims): regularised toward N(0, I) by an
        adversarial discriminator (see AAEDiscriminator).  Captures pose,
        lighting, expression — everything that is not identity.
      - z_id   (remaining dims): supervised to match the frozen ArcFace
        embedding of the input face (LatentIdentityLoss).  Because ArcFace
        embeddings generalise to unseen people, so does this half.

    The decoder consumes the full concatenated latent.

    Input/output: (B, 3, image_size, image_size) in [-1, 1].
    """

    def __init__(self, base_ch: int = 64, num_res: int = 2,
                 latent_dim: int = 1024, style_dim: int = 512,
                 image_size: int = 256):
        super().__init__()
        assert 0 < style_dim < latent_dim, "style_dim must be in (0, latent_dim)"
        self.latent_dim = latent_dim
        self.style_dim  = style_dim
        self.id_dim     = latent_dim - style_dim
        self.encoder = Encoder(base_ch, num_res, latent_dim, image_size)
        self.decoder = Decoder(base_ch, num_res, latent_dim, image_size)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def split(self, z: torch.Tensor):
        """Return (z_style, z_id)."""
        return z[:, :self.style_dim], z[:, self.style_dim:]

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        z_style, z_id = self.split(z)
        recon = self.decode(z)
        return recon, z, (z_style, z_id)


class AAEDiscriminator(nn.Module):
    """
    Adversarial discriminator over the style latent z_style.  Trained to tell a
    true N(0, I) sample from the encoder's z_style; the encoder is trained to
    fool it, which pushes the aggregated posterior of z_style toward N(0, I)
    without injecting noise into the model (the AAE advantage over a VAE).

    Outputs raw logits (use with BCEWithLogitsLoss).
    """

    def __init__(self, style_dim: int = 512, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(style_dim, hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden, hidden // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, z_style: torch.Tensor) -> torch.Tensor:
        return self.net(z_style)
