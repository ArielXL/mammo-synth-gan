import argparse, os, glob, random, math
from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd
import numpy as np
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import LambdaLR


# ---------- Utilidades ----------
def emoji_print(msg: str):
    print(msg, flush=True)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def print_progress_bar(current: int, total: int, prefix: str = "", length: int = 50):
    if total <= 0:
        return
    fraction = min(max(current / float(total), 0.0), 1.0)
    filled = int(length * fraction)
    bar = "█" * filled + "─" * (length - filled)
    percent = int(fraction * 100)
    msg = f"\r{prefix} |{bar}| {percent:3d}% ({current}/{total})"
    print(msg, end="", flush=True)
    if current >= total:
        print()


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def select_device(prefer_cuda_id: int = 0):
    try:
        if torch.cuda.is_available():
            if prefer_cuda_id < torch.cuda.device_count():
                dev = torch.device(f"cuda:{prefer_cuda_id}")
                _ = torch.cuda.get_device_name(dev)
                return (
                    dev,
                    f"GPU CUDA:{prefer_cuda_id} - " + torch.cuda.get_device_name(dev),
                )
            dev = torch.device("cuda:0")
            return dev, f"GPU CUDA:0 - " + torch.cuda.get_device_name(dev)
    except Exception:
        pass
    try:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps"), "Apple MPS (Metal)"
    except Exception:
        pass
    return torch.device("cpu"), "CPU"


def list_pair_folders_from_excel(
    excel_path: str, data_root: str, start_index: int, review_accept_value: str
) -> List[str]:
    df = pd.read_excel(excel_path)
    if "review" not in df.columns:
        raise ValueError("La columna 'review' no existe en pairs.xlsx")
    df = df[
        df["review"].astype(str).str.upper() == review_accept_value.upper()
    ].reset_index(drop=True)
    folders = []
    for i in range(len(df)):
        idx = start_index + i
        folder = os.path.join(data_root, f"{idx:04d}")
        b_mask = os.path.join(folder, "before_image_mask.png")
        a_mask = os.path.join(folder, "after_image_mask.png")
        if os.path.isdir(folder) and os.path.isfile(b_mask) and os.path.isfile(a_mask):
            folders.append(folder)
        else:
            emoji_print(f"⚠️  Saltando carpeta faltante o incompleta: {folder}")
    return folders


def split_train_test(
    items: List[str], train_ratio: float, seed: int
) -> Tuple[List[str], List[str]]:
    rnd = random.Random(seed)
    perm = items[:]
    rnd.shuffle(perm)
    n_train = int(round(len(perm) * train_ratio))
    return perm[:n_train], perm[n_train:]


def pil_to_tensor(img: Image.Image, size: int) -> torch.Tensor:
    if img.mode != "L":
        img = img.convert("L")
    if img.size != (size, size):
        img = img.resize((size, size), resample=Image.LANCZOS)
    arr = np.array(img).astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).unsqueeze(0)


def tensor_to_pil_gray(t: torch.Tensor) -> Image.Image:
    t = t.detach().cpu().clamp_(-1, 1)
    arr = ((t.squeeze(0) + 1.0) * 127.5).round().numpy().astype(np.uint8)
    return Image.fromarray(arr).convert("L")


# ---------- Métricas diferenciables (SSIM & TV) ----------
class SSIMLoss(nn.Module):
    """SSIM simple (grayscale) usando ventana 3x3 promedio. Compatible con GPU."""

    def __init__(self, C1=0.01**2, C2=0.03**2):
        super().__init__()
        self.C1 = C1
        self.C2 = C2
        kernel = torch.ones((1, 1, 3, 3), dtype=torch.float32) / 9.0
        self.register_buffer("kernel", kernel)

    def forward(self, x, y):
        k = self.kernel
        if k.device != x.device or k.dtype != x.dtype:
            k = k.to(device=x.device, dtype=x.dtype)
        mu_x = torch.nn.functional.conv2d(x, k, padding=1)
        mu_y = torch.nn.functional.conv2d(y, k, padding=1)
        sigma_x = torch.nn.functional.conv2d(x * x, k, padding=1) - mu_x * mu_x
        sigma_y = torch.nn.functional.conv2d(y * y, k, padding=1) - mu_y * mu_y
        sigma_xy = torch.nn.functional.conv2d(x * y, k, padding=1) - mu_x * mu_y
        C1, C2 = self.C1, self.C2
        ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / (
            (mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2) + 1e-8
        )
        return 1.0 - ssim_map.mean()


def tv_loss(x):
    return (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean() + (
        x[:, :, 1:, :] - x[:, :, :-1, :]
    ).abs().mean()


# ---------- Dataset ----------
class MaskPairsDataset(Dataset):
    def __init__(self, folders: List[str], size: int = 256):
        self.folders = folders
        self.size = size

    def __len__(self):
        return len(self.folders)

    def _find_after_image_path(self, folder):
        for ext in ["png", "jpg", "jpeg", "bmp", "tiff"]:
            cand = os.path.join(folder, f"after_image.{ext}")
            if os.path.isfile(cand):
                return cand
        return None

    def _find_before_image_path(self, folder):
        for ext in ["png", "jpg", "jpeg", "bmp", "tiff"]:
            cand = os.path.join(folder, f"before_image.{ext}")
            if os.path.isfile(cand):
                return cand
        return None

    def __getitem__(self, idx):
        folder = self.folders[idx]
        b_mask = os.path.join(folder, "before_image_mask.png")
        a_mask = os.path.join(folder, "after_image_mask.png")
        x_mask = pil_to_tensor(Image.open(b_mask).convert("L"), self.size)
        y_mask = pil_to_tensor(Image.open(a_mask).convert("L"), self.size)

        a_img_path = self._find_after_image_path(folder)
        y_img = (
            pil_to_tensor(Image.open(a_img_path).convert("L"), self.size)
            if a_img_path
            else y_mask
        )

        b_img_path = self._find_before_image_path(folder)
        x_img = (
            pil_to_tensor(Image.open(b_img_path).convert("L"), self.size)
            if b_img_path
            else x_mask
        )

        return x_mask, y_mask, y_img, x_img, os.path.basename(folder)


# ---------- Modelos (Pix2Pix) ----------
class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, down=True, act="leaky", use_dropout=False):
        super().__init__()
        if down:
            layers = [
                nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_ch),
            ]
        else:
            layers = [
                nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_ch),
            ]
        self.block = nn.Sequential(*layers)
        self.activation = nn.LeakyReLU(0.2, True) if act == "leaky" else nn.ReLU(True)
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.block(x)
        x = self.activation(x)
        if self.use_dropout:
            x = self.dropout(x)
        return x


class UNetGenerator(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, nf=64):
        super().__init__()
        self.e1 = nn.Sequential(nn.Conv2d(in_ch, nf, 4, 2, 1), nn.LeakyReLU(0.2, True))
        self.e2 = UNetBlock(nf, nf * 2, down=True, act="leaky")
        self.e3 = UNetBlock(nf * 2, nf * 4, down=True, act="leaky")
        self.e4 = UNetBlock(nf * 4, nf * 8, down=True, act="leaky")
        self.e5 = UNetBlock(nf * 8, nf * 8, down=True, act="leaky")
        self.e6 = UNetBlock(nf * 8, nf * 8, down=True, act="leaky")
        self.e7 = UNetBlock(nf * 8, nf * 8, down=True, act="leaky")
        self.e8 = nn.Sequential(
            nn.Conv2d(nf * 8, nf * 8, 4, 2, 1, bias=False), nn.ReLU(True)
        )
        self.d1 = UNetBlock(nf * 8, nf * 8, down=False, act="relu", use_dropout=True)
        self.d2 = UNetBlock(
            nf * 8 * 2, nf * 8, down=False, act="relu", use_dropout=True
        )
        self.d3 = UNetBlock(
            nf * 8 * 2, nf * 8, down=False, act="relu", use_dropout=True
        )
        self.d4 = UNetBlock(nf * 8 * 2, nf * 8, down=False, act="relu")
        self.d5 = UNetBlock(nf * 8 * 2, nf * 4, down=False, act="relu")
        self.d6 = UNetBlock(nf * 4 * 2, nf * 2, down=False, act="relu")
        self.d7 = UNetBlock(nf * 2 * 2, nf, down=False, act="relu")
        self.d8 = nn.ConvTranspose2d(nf * 2, out_ch, 4, 2, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)
        e6 = self.e6(e5)
        e7 = self.e7(e6)
        e8 = self.e8(e7)
        d1 = self.d1(e8)
        d2 = self.d2(torch.cat([d1, e7], dim=1))
        d3 = self.d3(torch.cat([d2, e6], dim=1))
        d4 = self.d4(torch.cat([d3, e5], dim=1))
        d5 = self.d5(torch.cat([d4, e4], dim=1))
        d6 = self.d6(torch.cat([d5, e3], dim=1))
        d7 = self.d7(torch.cat([d6, e2], dim=1))
        d8 = self.d8(torch.cat([d7, e1], dim=1))
        return self.tanh(d8)


class PatchDiscriminator(nn.Module):
    def __init__(self, in_ch=2, nf=64):
        super().__init__()
        self.c1 = nn.Sequential(nn.Conv2d(in_ch, nf, 4, 2, 1), nn.LeakyReLU(0.2, True))
        self.c2 = nn.Sequential(
            nn.Conv2d(nf, nf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nf * 2),
            nn.LeakyReLU(0.2, True),
        )
        self.c3 = nn.Sequential(
            nn.Conv2d(nf * 2, nf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nf * 4),
            nn.LeakyReLU(0.2, True),
        )
        self.c4 = nn.Sequential(
            nn.Conv2d(nf * 4, nf * 8, 4, 1, 1, bias=False),
            nn.BatchNorm2d(nf * 8),
            nn.LeakyReLU(0.2, True),
        )
        self.c5 = nn.Conv2d(nf * 8, 1, 4, 1, 1)

    def forward(self, x, y):
        z = torch.cat([x, y], dim=1)
        h1 = self.c1(z)
        h2 = self.c2(h1)
        h3 = self.c3(h2)
        h4 = self.c4(h3)
        out = self.c5(h4)
        return out


# ---------- Checkpointing ----------
@dataclass
class TrainState:
    epoch: int
    G_state: dict
    D_state: dict
    optG_state: dict
    optD_state: dict
    args: dict


def save_checkpoint(state: TrainState, ckpt_dir: str, epoch: int):
    ensure_dir(ckpt_dir)
    path = os.path.join(ckpt_dir, f"checkpoint_epoch_{epoch:04d}.pt")
    # torch.save(
    #     {
    #         "epoch": state.epoch,
    #         "G": state.G_state,
    #         "D": state.D_state,
    #         "optG": state.optG_state,
    #         "optD": state.optD_state,
    #         "args": state.args,
    #     },
    #     path,
    # )
    emoji_print(f"💾 Checkpoint guardado: {path}")


def find_penultimate_checkpoint(ckpt_dir: str):
    files = sorted(glob.glob(os.path.join(ckpt_dir, "checkpoint_epoch_*.pt")))
    if len(files) >= 2:
        return files[-2], files[-1]
    return None, None


def write_hparams_txt(out_dir: str, args, extra: dict):
    text_path = os.path.join(out_dir, "hparams.txt")
    with open(text_path, "w", encoding="utf-8") as f:
        f.write("# Hiperparámetros y estado de entrenamiento\n")
        for k, v in sorted(vars(args).items()):
            f.write(f"{k}: {v}\n")
        if extra:
            f.write("\n# Info adicional\n")
            for k, v in extra.items():
                f.write(f"{k}: {v}\n")
    emoji_print(f"📝 Hiperparámetros guardados en: {text_path}")


# ---------- Gráficas ----------
def plot_losses_curves(history, out_dir):
    ensure_dir(out_dir)
    epochs = history["epoch"]
    G_losses = history["G"]
    D_losses = history["D"]
    G_adv_losses = history["G_adv"]
    G_L1_losses = history["G_L1"]

    def save_plot(y_values, title, filename, labels=None):
        plt.figure()
        if isinstance(y_values, list) and labels is None:
            plt.plot(epochs, y_values)
        elif isinstance(y_values, list) and labels is not None:
            for y, lbl in zip(y_values, labels):
                plt.plot(epochs, y, label=lbl)
            plt.legend()
        plt.xlabel("Épocas")
        plt.ylabel("Pérdida")
        plt.title(title)
        plt.tight_layout()
        path = os.path.join(out_dir, filename)
        plt.savefig(path, dpi=150)
        plt.close()
        emoji_print(f"📈 Gráfica guardada: {path}")

    save_plot(
        [G_losses, D_losses, G_adv_losses, G_L1_losses],
        "Pérdidas: G, D, G_adv, G_L1",
        "loss_all.png",
        labels=["G", "D", "G_adv", "G_L1"],
    )
    save_plot(G_losses, "Pérdida del Generador (G)", "loss_G.png")
    save_plot(D_losses, "Pérdida del Discriminador (D)", "loss_D.png")
    save_plot(
        G_adv_losses, "Pérdida Adversarial del Generador (G_adv)", "loss_G_adv.png"
    )
    save_plot(G_L1_losses, "Pérdida L1 del Generador (G_L1)", "loss_G_L1.png")
    save_plot([G_losses, D_losses], "G vs D", "loss_G_vs_D.png", labels=["G", "D"])


# ---------- Test: guardar resultados por carpeta (mismo nombre que en data-root) ----------
def save_test_triplets(G, dataloader, device, out_dir, enable_stage2=False, G2=None):
    ensure_dir(out_dir)
    G.eval()
    if G2 is not None:
        G2.eval()
    with torch.no_grad():
        for _, (
            x_mask_before,
            y_mask_after,
            y_img_after,
            x_img_before,
            folder_name,
        ) in enumerate(dataloader, start=1):
            x = x_mask_before.to(device)
            y = y_mask_after.to(device)
            y_img_after = y_img_after.to(device)
            x_img_before = x_img_before.to(device)

            fake_mask = G(x)

            before_mask_pil = tensor_to_pil_gray(x[0])
            gen_mask_pil = tensor_to_pil_gray(fake_mask[0])
            after_mask_pil = tensor_to_pil_gray(y[0])
            before_img_pil = tensor_to_pil_gray(x_img_before[0])
            after_img_pil = tensor_to_pil_gray(y_img_after[0])

            pair_folder = os.path.join(
                out_dir, folder_name[0]
            )  # MISMO nombre que en ../../images
            ensure_dir(pair_folder)

            before_mask_pil.save(
                os.path.join(pair_folder, "before_image_mask.png"),
                format="PNG",
                optimize=True,
                compress_level=6,
            )
            gen_mask_pil.save(
                os.path.join(pair_folder, "generated_mask.png"),
                format="PNG",
                optimize=True,
                compress_level=6,
            )
            after_mask_pil.save(
                os.path.join(pair_folder, "after_image_mask.png"),
                format="PNG",
                optimize=True,
                compress_level=6,
            )
            before_img_pil.save(
                os.path.join(pair_folder, "before_image.png"),
                format="PNG",
                optimize=True,
                compress_level=6,
            )
            after_img_pil.save(
                os.path.join(pair_folder, "after_image.png"),
                format="PNG",
                optimize=True,
                compress_level=6,
            )

            if enable_stage2 and G2 is not None:
                mammo = G2(fake_mask)
                mammo_pil = tensor_to_pil_gray(mammo[0])
                mammo_pil.save(
                    os.path.join(pair_folder, "generated_mammo.png"),
                    format="PNG",
                    optimize=True,
                    compress_level=6,
                )


# ---------- Entrenamiento ----------
def train(args):
    device, device_name = select_device(prefer_cuda_id=args.gpu_id)
    emoji_print(f"🖥️  Dispositivo seleccionado: {device_name}")
    emoji_print("🚀 Inicio del pipeline de entrenamiento")
    emoji_print("🧩 Configurando datos y lectura de Excel...")

    folders = list_pair_folders_from_excel(
        args.excel, args.data_root, args.start_index, args.review_accept_value
    )
    emoji_print(
        f"📄 Excel: {args.excel} | data_root: {args.data_root} | start_index: {args.start_index}"
    )
    if len(folders) == 0:
        raise RuntimeError(
            "No se encontraron carpetas válidas para entrenamiento con review == 'ACEPTAR'."
        )

    train_folders, test_folders = split_train_test(
        folders, train_ratio=args.train_ratio, seed=args.seed
    )
    emoji_print(
        f"📦 Total pares ACEPTADOS: {len(folders)} | Entrenamiento: {len(train_folders)} | Prueba: {len(test_folders)}"
    )
    emoji_print("🧰 Construyendo datasets y dataloaders...")

    train_ds = MaskPairsDataset(train_folders, size=args.size)
    test_ds = MaskPairsDataset(test_folders, size=args.size)
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)
    emoji_print(
        f"🛠️  DataLoader listo ➜ batch_size={args.batch_size}, workers={args.workers}"
    )

    # Modelos
    G = UNetGenerator(in_ch=1, out_ch=1, nf=args.ngf).to(device)
    D = PatchDiscriminator(in_ch=2, nf=args.ndf).to(device)
    emoji_print(
        f"🏗️  Modelos creados ➜ G(UNet nf={args.ngf}), D(PatchGAN nf={args.ndf})"
    )
    G2 = D2 = None
    if args.enable_mammo_synthesis:
        G2 = UNetGenerator(in_ch=1, out_ch=1, nf=args.ngf).to(device)
        D2 = PatchDiscriminator(in_ch=2, nf=args.ndf).to(device)
        emoji_print(
            f"🏗️  Etapa 2 activa ➜ G2(UNet nf={args.ngf}), D2(PatchGAN nf={args.ndf})"
        )

    # Pérdidas & Optimizadores
    GANLoss = nn.MSELoss if args.gan_loss == "ls" else nn.BCEWithLogitsLoss
    criterion_GAN = GANLoss()
    criterion_L1 = nn.L1Loss()
    ssim_fn = SSIMLoss()

    optG = torch.optim.Adam(G.parameters(), lr=args.lrG, betas=(0.5, 0.999))
    optD = torch.optim.Adam(D.parameters(), lr=args.lrD, betas=(0.5, 0.999))
    if args.enable_mammo_synthesis:
        optG2 = torch.optim.Adam(G2.parameters(), lr=args.lrG2, betas=(0.5, 0.999))
        optD2 = torch.optim.Adam(D2.parameters(), lr=args.lrD2, betas=(0.5, 0.999))

    # ---------- Schedulers coseno en antifase (LambdaLR) ----------
    schG = schD = schG2 = schD2 = None
    if args.use_cosine:
        T = max(1, args.cosine_Tmax)
        m = float(args.cosine_minlr_mult)

        def cos_mult(epoch, shift=0.0):
            phase = (epoch + shift) % T
            return m + (1.0 - m) * 0.5 * (1.0 + math.cos(math.pi * (phase / T)))

        schG = LambdaLR(optG, lr_lambda=lambda ep: cos_mult(ep, shift=0.0))
        schD = LambdaLR(optD, lr_lambda=lambda ep: cos_mult(ep, shift=T / 2.0))
        if args.enable_mammo_synthesis:
            schG2 = LambdaLR(optG2, lr_lambda=lambda ep: cos_mult(ep, shift=0.0))
            schD2 = LambdaLR(optD2, lr_lambda=lambda ep: cos_mult(ep, shift=T / 2.0))

    # Checkpoints
    ckpt_dir = os.path.join(args.output, "checkpoints")
    ensure_dir(ckpt_dir)

    # Reanudar desde el penúltimo checkpoint
    start_epoch = 1
    if not args.no_resume:
        penult, last = find_penultimate_checkpoint(ckpt_dir)
        if last is not None:
            emoji_print(f"🔁 Reanudando desde el checkpoint: {last}")
            state = torch.load(last, map_location=device)
            G.load_state_dict(state["G"])
            D.load_state_dict(state["D"])
            optG.load_state_dict(state["optG"])
            optD.load_state_dict(state["optD"])
            start_epoch = int(state["epoch"]) + 1
        else:
            emoji_print(
                "ℹ️  No hay al menos dos checkpoints. Se empieza desde la época 1."
            )

    # Salidas & hparams
    ensure_dir(args.output)
    ensure_dir(os.path.join(args.output, "plots"))
    ensure_dir(os.path.join(args.output, "test_results"))
    emoji_print(f"📁 Directorios de salida listos en: {args.output}")
    emoji_print("📝 Guardando hiperparámetros.")
    G_params, G_trainable = count_params(G)
    D_params, D_trainable = count_params(D)
    extra_info = {
        "device": str(device),
        "device_name": device_name,
        "start_epoch": start_epoch,
        "train_pairs": len(train_folders),
        "test_pairs": len(test_folders),
        "total_pairs_aceptados": len(folders),
        "checkpoint_every": args.checkpoint_every,
        "resume_enabled": (not args.no_resume),
        "ngf": args.ngf,
        "ndf": args.ndf,
        "lrG": args.lrG,
        "lrD": args.lrD,
        "lambda_L1": args.lambda_L1,
        "gan_loss": args.gan_loss,
        "label_smoothing": args.label_smoothing,
        "instance_noise_start": args.instance_noise,
        "instance_noise_end": args.instance_noise_end,
        "use_cosine": args.use_cosine,
        "cosine_Tmax": args.cosine_Tmax,
        "cosine_minlr_mult": args.cosine_minlr_mult,
        "n_critic": args.n_critic,
        "n_critic_strong": args.n_critic_strong,
        "alt_critic_period": args.alt_critic_period,
        "G_params_total": G_params,
        "G_params_trainable": G_trainable,
        "D_params_total": D_params,
        "D_params_trainable": D_trainable,
    }
    if args.enable_mammo_synthesis:
        G2_params, G2_trainable = count_params(G2)
        D2_params, D2_trainable = count_params(D2)
        extra_info.update(
            {
                "enable_mammo_synthesis": True,
                "lambda_L1_img": args.lambda_L1_img,
                "lambda_SSIM": args.lambda_SSIM,
                "lambda_TV": args.lambda_TV,
                "lrG2": args.lrG2,
                "lrD2": args.lrD2,
                "G2_params_total": G2_params,
                "G2_params_trainable": G2_trainable,
                "D2_params_total": D2_params,
                "D2_params_trainable": D2_trainable,
            }
        )
    write_hparams_txt(args.output, args, extra_info)

    # Historia de pérdidas
    history = {"epoch": [], "G": [], "D": [], "G_adv": [], "G_L1": []}
    if args.enable_mammo_synthesis:
        history_img = {"epoch": [], "G2": [], "D2": [], "G2_adv": [], "G2_L1": []}

    # Entrenamiento
    for epoch in range(start_epoch, args.epochs + 1):
        emoji_print(f"⏳ Comenzando época {epoch}/{args.epochs}...")
        G.train()
        D.train()
        if args.enable_mammo_synthesis and G2 is not None and D2 is not None:
            G2.train()
            D2.train()

        # --- Alternancia de fases (n-critic) ---
        phase_len = max(1, args.alt_critic_period)
        phase = (epoch - 1) // phase_len
        d_strong = phase % 2 == 0
        n_critic_now = args.n_critic_strong if d_strong else args.n_critic
        emoji_print(
            f"🔁 Fase: {'D-fuerte' if d_strong else 'G-fuerte'} | n_critic={n_critic_now}"
        )

        epoch_G_loss = epoch_D_loss = epoch_G_adv = epoch_G_L1 = 0.0
        if args.enable_mammo_synthesis:
            epoch_G2_loss = epoch_D2_loss = epoch_G2_adv = epoch_G2_L1 = 0.0

        total_batches = max(1, len(train_loader))
        batch_idx = 0

        # Instance noise
        if args.instance_noise > 0.0:
            sigma = max(
                args.instance_noise_end,
                args.instance_noise * (1.0 - (epoch - 1) / max(1, args.epochs)),
            )
        else:
            sigma = 0.0

        for x_mask_before, y_mask_after, y_img_after, x_img_before, _ in train_loader:
            batch_idx += 1
            x = x_mask_before.to(device, non_blocking=True)
            y = y_mask_after.to(device, non_blocking=True)
            y_img = y_img_after.to(device, non_blocking=True)

            # Label smoothing (o no)
            if args.label_smoothing:
                real_t = 0.9 + 0.1 * random.random()
                fake_t = 0.0 + 0.1 * random.random()
            else:
                real_t, fake_t = 1.0, 0.0

            # ----- Discriminador: n_critic_now pasos -----
            for _ in range(n_critic_now):
                optD.zero_grad()
                y_in = y + (torch.randn_like(y) * sigma if sigma > 0 else 0)
                out_real = D(x, y_in)
                loss_D_real = criterion_GAN(
                    out_real, torch.full_like(out_real, real_t, device=device)
                )

                with torch.no_grad():
                    fake_y = G(x)
                fake_in = fake_y + (
                    torch.randn_like(fake_y) * sigma if sigma > 0 else 0
                )
                out_fake = D(x, fake_in.detach())
                loss_D_fake = criterion_GAN(
                    out_fake, torch.full_like(out_fake, fake_t, device=device)
                )
                loss_D = 0.5 * (loss_D_real + loss_D_fake)
                loss_D.backward()
                optD.step()

            # ----- Generador -----
            g_steps = 1
            for _ in range(g_steps):
                optG.zero_grad()
                fake_y = G(x)
                out_fake_for_G = D(x, fake_y)
                loss_G_adv = criterion_GAN(
                    out_fake_for_G,
                    torch.full_like(out_fake_for_G, real_t, device=device),
                )
                loss_G_L1 = criterion_L1(fake_y, y) * args.lambda_L1
                loss_G = loss_G_adv + loss_G_L1
                loss_G.backward()
                optG.step()

            epoch_G_loss += loss_G.item()
            epoch_D_loss += loss_D.item()
            epoch_G_adv += loss_G_adv.item()
            epoch_G_L1 += loss_G_L1.item()

            # ----- Etapa 2 (opcional): máscara → mamografía -----
            if args.enable_mammo_synthesis and G2 is not None and D2 is not None:
                # D2
                optD2.zero_grad()
                out_real2 = D2(y, y_img)
                loss_D2_real = criterion_GAN(
                    out_real2, torch.full_like(out_real2, real_t, device=device)
                )
                with torch.no_grad():
                    fake_img = G2(y)
                out_fake2 = D2(y, fake_img.detach())
                loss_D2_fake = criterion_GAN(
                    out_fake2, torch.full_like(out_fake2, fake_t, device=device)
                )
                loss_D2 = 0.5 * (loss_D2_real + loss_D2_fake)
                loss_D2.backward()
                optD2.step()

                # G2
                optG2.zero_grad()
                fake_img = G2(y)
                out_fake_for_G2 = D2(y, fake_img)
                loss_G2_adv = criterion_GAN(
                    out_fake_for_G2,
                    torch.full_like(out_fake_for_G2, real_t, device=device),
                )
                l1 = criterion_L1(fake_img, y_img) * args.lambda_L1_img
                ssim = ssim_fn((fake_img + 1) / 2, (y_img + 1) / 2) * args.lambda_SSIM
                tv = tv_loss(fake_img) * args.lambda_TV
                loss_G2 = loss_G2_adv + l1 + ssim + tv
                loss_G2.backward()
                optG2.step()

                epoch_G2_loss += loss_G2.item()
                epoch_D2_loss += loss_D2.item()
                epoch_G2_adv += loss_G2_adv.item()
                epoch_G2_L1 += l1.item()

            # Progreso
            if args.progress and (
                batch_idx % max(1, args.progress_every) == 0
                or batch_idx == total_batches
            ):
                print_progress_bar(
                    batch_idx,
                    total_batches,
                    prefix=f"📦 Época {epoch}/{args.epochs}",
                    length=args.progress_length,
                )

        # Promedios por época
        n_batches = max(1, len(train_loader))
        epoch_G_loss /= n_batches
        epoch_D_loss /= n_batches
        epoch_G_adv /= n_batches
        epoch_G_L1 /= n_batches
        if args.enable_mammo_synthesis:
            epoch_G2_loss /= n_batches
            epoch_D2_loss /= n_batches
            epoch_G2_adv /= n_batches
            epoch_G2_L1 /= n_batches

        history["epoch"].append(epoch)
        history["G"].append(epoch_G_loss)
        history["D"].append(epoch_D_loss)
        history["G_adv"].append(epoch_G_adv)
        history["G_L1"].append(epoch_G_L1)
        if args.enable_mammo_synthesis:
            history_img["epoch"].append(epoch)
            history_img["G2"].append(epoch_G2_loss)
            history_img["D2"].append(epoch_D2_loss)
            history_img["G2_adv"].append(epoch_G2_adv)
            history_img["G2_L1"].append(epoch_G2_L1)

        msg = f"🧪 Época {epoch}/{args.epochs} ➜ G: {epoch_G_loss:.4f} | D: {epoch_D_loss:.4f} | G_adv: {epoch_G_adv:.4f} | G_L1: {epoch_G_L1:.4f}"
        if args.enable_mammo_synthesis:
            msg += f" || G2: {epoch_G2_loss:.4f} | D2: {epoch_D2_loss:.4f} | G2_adv: {epoch_G2_adv:.4f} | G2_L1: {epoch_G2_L1:.4f}"
        emoji_print(msg)

        # Checkpoint
        if (epoch % args.checkpoint_every) == 0:
            state = TrainState(
                epoch=epoch,
                G_state=G.state_dict(),
                D_state=D.state_dict(),
                optG_state=optG.state_dict(),
                optD_state=optD.state_dict(),
                args=vars(args),
            )
            save_checkpoint(state, ckpt_dir, epoch)
        else:
            emoji_print("💡 (Sin checkpoint en esta época)")

        # Step de LR coseno
        if args.use_cosine:
            if schG is not None:
                schG.step()
            if schD is not None:
                schD.step()
            if args.enable_mammo_synthesis and schG2 is not None and schD2 is not None:
                schG2.step()
                schD2.step()

    # Gráficas + Test
    plot_losses_curves(history, os.path.join(args.output, "plots"))
    if (
        args.enable_mammo_synthesis
        and "history_img" in locals()
        and len(history_img["epoch"]) > 0
    ):
        tmp = {
            "epoch": history_img["epoch"],
            "G": history_img["G2"],
            "D": history_img["D2"],
            "G_adv": history_img["G2_adv"],
            "G_L1": history_img["G2_L1"],
        }
        plot_losses_curves(tmp, os.path.join(args.output, "plots_stage2"))

    emoji_print("🖼️  Generando y guardando gráficas de pérdidas...")
    emoji_print("🧪 Generando resultados de prueba...")
    emoji_print(f"🧪 Evaluación en test: {len(test_ds)} pares")
    save_test_triplets(
        G,
        test_loader,
        device,
        os.path.join(args.output, "test_results"),
        enable_stage2=args.enable_mammo_synthesis,
        G2=(G2 if args.enable_mammo_synthesis else None),
    )
    emoji_print("✅ Entrenamiento finalizado.")
    emoji_print(f"🧾 Hiperparámetros: {os.path.join(args.output, 'hparams.txt')}")
    emoji_print(f"📊 Gráficas: {os.path.join(args.output, 'plots')}")
    emoji_print(f"🖼️ Resultados de prueba: {os.path.join(args.output, 'test_results')}")


# ---------- Args ----------
def parse_args():
    p = argparse.ArgumentParser(
        description="Entrena una cGAN (Pix2Pix) sobre pares de máscaras para pronosticar evolución tumoral."
    )
    # Datos
    p.add_argument(
        "--excel",
        default="../pairs.xlsx",
        help="Ruta a pairs.xlsx (debe contener columna 'review').",
    )
    p.add_argument(
        "--data-root",
        default="../../images",
        help="Raíz con carpetas 0001, 0002, ... con las máscaras e imágenes.",
    )
    p.add_argument(
        "--start-index",
        type=int,
        default=1,
        help="Índice inicial usado al crear carpetas (0001=1).",
    )
    p.add_argument("--output", default="../runs", help="Directorio de salida.")
    p.add_argument(
        "--review-accept-value",
        default="ACEPTAR",
        help="Valor de 'review' que indica par válido.",
    )
    p.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Proporción train/test (0.8=80/20).",
    )
    p.add_argument("--seed", type=int, default=1337, help="Semilla reproducible.")
    # Modelo / entrenamiento
    p.add_argument("--size", type=int, default=256, help="Tamaño de entrada/salida.")
    p.add_argument("--epochs", type=int, default=200, help="Épocas.")
    p.add_argument("--batch-size", type=int, default=1, help="Batch size.")
    p.add_argument("--workers", type=int, default=0, help="Dataloader workers.")
    p.add_argument("--ngf", type=int, default=64, help="Canales base del generador.")
    p.add_argument(
        "--ndf", type=int, default=64, help="Canales base del discriminador."
    )
    # TTUR por defecto: lrD > lrG
    p.add_argument("--lrG", type=float, default=2e-4, help="LR generador.")
    p.add_argument("--lrD", type=float, default=4e-4, help="LR discriminador.")
    p.add_argument(
        "--lambda-L1",
        dest="lambda_L1",
        type=float,
        default=100.0,
        help="Peso L1 (Isola et al.).",
    )
    # Estabilizadores
    p.add_argument(
        "--gan-loss",
        choices=["bce", "ls"],
        default="ls",
        help="Tipo de pérdida adversarial.",
    )
    g = p.add_mutually_exclusive_group()
    g.add_argument(
        "--label-smoothing",
        dest="label_smoothing",
        action="store_true",
        help="Usar smoothing en etiquetas reales/falsas.",
    )
    g.add_argument(
        "--no-label-smoothing",
        dest="label_smoothing",
        action="store_false",
        help="Desactivar smoothing en etiquetas.",
    )
    p.set_defaults(label_smoothing=True)
    p.add_argument(
        "--instance-noise",
        type=float,
        default=0.05,
        help="Ruido gaussiano inicial en entradas de D (decae).",
    )
    p.add_argument("--instance-noise-end", type=float, default=0.0, help="Ruido final.")
    # Etapa 2 (máscara → mamografía)
    p.add_argument(
        "--enable-mammo-synthesis",
        action="store_true",
        default=True,
        help="Etapa 2 activa.",
    )
    p.add_argument(
        "--lambda-L1-img", type=float, default=50.0, help="Peso L1 para etapa 2."
    )
    p.add_argument(
        "--lambda-SSIM",
        dest="lambda_SSIM",
        type=float,
        default=5.0,
        help="Peso SSIM para etapa 2.",
    )
    p.add_argument(
        "--lambda-TV",
        dest="lambda_TV",
        type=float,
        default=1e-4,
        help="Peso TV para etapa 2.",
    )
    p.add_argument("--lrG2", type=float, default=2e-4, help="LR del generador etapa 2.")
    p.add_argument(
        "--lrD2", type=float, default=4e-4, help="LR del discriminador etapa 2."
    )
    # n-critic alternante y LR coseno
    p.add_argument(
        "--n-critic",
        type=int,
        default=1,
        help="Pasos de D por cada paso de G (fase G-fuerte).",
    )
    p.add_argument(
        "--n-critic-strong",
        type=int,
        default=3,
        help="Pasos de D por cada paso de G en fase D-fuerte.",
    )
    p.add_argument(
        "--alt-critic-period",
        type=int,
        default=10,
        help="Épocas por fase antes de alternar (D-fuerte <-> G-fuerte).",
    )
    p.add_argument(
        "--use-cosine",
        action="store_true",
        default=True,
        help="Activar scheduler coseno (LambdaLR).",
    )
    p.add_argument(
        "--cosine-Tmax", type=int, default=20, help="Período (épocas) del coseno."
    )
    p.add_argument(
        "--cosine-minlr-mult",
        type=float,
        default=0.1,
        help="Mínimo LR como múltiplo del LR base.",
    )
    # Checkpoints, dispositivo, progreso
    p.add_argument(
        "--checkpoint-every",
        type=int,
        default=20,
        help="Guardar checkpoint cada N épocas.",
    )
    p.add_argument(
        "--no-resume",
        action="store_true",
        help="No reanudar desde el penúltimo checkpoint.",
    )
    p.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="Índice de GPU CUDA preferida (por defecto 0).",
    )
    p.add_argument(
        "--progress",
        action="store_true",
        default=True,
        help="Mostrar barra de progreso por época.",
    )
    p.add_argument(
        "--progress-length",
        type=int,
        default=50,
        help="Longitud de la barra de progreso.",
    )
    p.add_argument(
        "--progress-every", type=int, default=1, help="Actualizar barra cada N batches."
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ensure_dir(args.output)
    train(args)
