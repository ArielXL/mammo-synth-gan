import os
import sys
import argparse

import numpy as np
import pandas as pd

from PIL import Image, ImageFile

from typing import Tuple

from utils.ensure_dir import ensure_dir
from utils.emoji_print import emoji_print


ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_image(path: str, mode: str = None) -> Image.Image | None:
    """
    SUMMARY
    -------
    Load an image from the given path. If mode is specified, convert to that mode.

    PARAMETERS
    ----------
    path : str
        Path to the image file.
    mode : str, optional
        Mode to convert the image to (e.g., 'RGB', 'L'). If None, keep original mode.

    RETURNS
    -------
    Image.Image | None
        Loaded PIL Image object, or None if loading failed.
    """
    try:
        img = Image.open(path)
        if mode:
            img = img.convert(mode)
        return img
    except Exception as e:
        emoji_print(f"💥 Error abriendo imagen: {path}\n   ➜ {e}")
        return None


def mask_to_bbox(mask_np: np.ndarray) -> Tuple[int, int, int, int] | None:
    """
    SUMMARY
    -------
    Compute bounding box of non-zero region in a binary mask.

    PARAMETERS
    ----------
    mask_np : np.ndarray
        2D numpy array representing the binary mask.

    RETURNS
    -------
    Tuple[int, int, int, int] | None
        Return bounding box (minx, miny, maxx, maxy) where mask > 0; None if empty.
    """
    ys, xs = np.where(mask_np > 0)
    if xs.size == 0 or ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def expand_square_bbox(
    bbox: Tuple[int, int, int, int], img_w: int, img_h: int, margin_ratio: float = 0.10
) -> Tuple[int, int, int, int]:
    """
    SUMMARY
    -------
    Expand a bounding box to be square with a margin, clamped to image bounds.

    PARAMETERS
    ----------
    bbox : Tuple[int, int, int, int]
        Original bounding box (x_min, y_min, x_max, y_max).
    img_w : int
        Width of the image.
    img_h : int
        Height of the image.
    margin_ratio : float, optional
        Margin ratio to add around the box (default is 0.10).

    RETURNS
    -------
    Tuple[int, int, int, int]
        Expanded bounding box (x_min, y_min, x_max, y_max) clamped to image bounds.
    """
    x_min, y_min, x_max, y_max = bbox
    w = x_max - x_min + 1
    h = y_max - y_min + 1

    side = max(w, h)
    cx = x_min + w / 2.0
    cy = y_min + h / 2.0

    side = int(round(side * (1.0 + margin_ratio)))

    new_x_min = int(round(cx - side / 2.0))
    new_y_min = int(round(cy - side / 2.0))
    new_x_max = new_x_min + side - 1
    new_y_max = new_y_min + side - 1

    if new_x_min < 0:
        shift = -new_x_min
        new_x_min += shift
        new_x_max += shift
    if new_y_min < 0:
        shift = -new_y_min
        new_y_min += shift
        new_y_max += shift
    if new_x_max >= img_w:
        shift = new_x_max - img_w + 1
        new_x_min -= shift
        new_x_max -= shift
    if new_y_max >= img_h:
        shift = new_y_max - img_h + 1
        new_y_min -= shift
        new_y_max -= shift

    new_x_min = max(0, new_x_min)
    new_y_min = max(0, new_y_min)
    new_x_max = min(img_w - 1, new_x_max)
    new_y_max = min(img_h - 1, new_y_max)

    return new_x_min, new_y_min, new_x_max, new_y_max


def crop_and_resize(
    img: Image.Image, bbox: Tuple[int, int, int, int], size: int
) -> Image.Image:
    """
    SUMMARY
    -------
    Crop the image to the given bounding box and resize to the specified size.

    PARAMETERS
    ----------
    img : Image.Image
        Input PIL Image.
    bbox : Tuple[int, int, int, int]
        Bounding box (x_min, y_min, x_max, y_max) to crop.
    size : int
        Size (width and height) to resize the cropped image to.

    RETURNS
    -------
    Image.Image
        Cropped and resized image.
    """
    x_min, y_min, x_max, y_max = bbox
    cropped = img.crop((x_min, y_min, x_max + 1, y_max + 1))
    resized = cropped.resize((size, size), resample=Image.LANCZOS)
    return resized


def save_image(
    img: Image.Image, path: str, image_format: str, jpeg_quality: int
) -> None:
    """
    SUMMARY
    -------
    Save the image to the specified path with given format and quality settings.

    PARAMETERS
    ----------
    img : Image.Image
        PIL Image to save.
    path : str
        Path to save the image.
    image_format : str
        Format to save the image in (e.g., 'png', 'jpg', 'jpeg').
    jpeg_quality : int
        Quality setting for JPEG format (1-100).

    RETURNS
    -------
    None
    """
    params = {}
    fmt = image_format.lower()
    if fmt in ["jpg", "jpeg"]:
        fmt = "JPEG"
        params["quality"] = int(jpeg_quality)
        params["optimize"] = True
        params["subsampling"] = 0
        params["progressive"] = True
    elif fmt == "png":
        fmt = "PNG"
        params["optimize"] = True
        params["compress_level"] = 6
    else:
        fmt = fmt.upper()
    img.save(path, format=fmt, **params)


def validate_columns(df: pd.DataFrame, cols: list) -> None:
    """
    SUMMARY
    -------
    Validate that the specified columns exist in the DataFrame.

    PARAMETERS
    ----------
    df : pd.DataFrame
        DataFrame to check.
    cols : list
        List of column names to validate.
    """
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise ValueError(f"Columnas faltantes en el Excel: {missing}")


def to_grayscale_mask(img: Image.Image) -> Image.Image:
    """
    SUMMARY
    -------
    Convert mask image to grayscale ('L') or keep binary ('1') if already so.

    PARAMETERS
    ----------
    img : Image.Image
        Input mask image.

    RETURNS
    -------
    Image.Image
        Grayscale or binary mask image.
    """
    if img.mode == "1":
        return img
    return img.convert("L")


# -------------------- Main processing --------------------


def process_row(
    idx_pair: int,
    row: pd.Series,
    out_dir: str,
    cols: dict,
    size: int,
    image_format: str,
    jpeg_quality: int,
    overwrite: bool,
    margin: float,
) -> bool:
    """
    SUMMARY
    -------
    Process a single row from the DataFrame: load images and masks, compute bounding boxes,
    crop, resize, and save them to the output directory.

    PARAMETERS
    ----------
    idx_pair : int
        Index of the pair (for folder naming).
    row : pd.Series
        DataFrame row containing image and mask paths.
    out_dir : str
        Output directory to save cropped pairs.
    cols : dict
        Dictionary mapping keys to column names in the DataFrame.
    size : int
        Size to resize cropped images to.
    image_format : str
        Format to save images in.
    jpeg_quality : int
        JPEG quality setting.
    overwrite : bool
        Whether to overwrite existing folders.
    margin : float
        Margin ratio to add around the bounding box.

    RETURNS
    -------
    bool
        True if processing was successful, False otherwise.
    """
    pair_folder = os.path.join(out_dir, f"{idx_pair:04d}")
    if (not overwrite) and os.path.exists(pair_folder) and os.listdir(pair_folder):
        emoji_print(
            f"⚠️  Saltando par #{idx_pair:04d}: carpeta ya existe y no está vacía ➜ {pair_folder}"
        )
        return True

    ensure_dir(pair_folder)

    b_img_path = str(row[cols["before_image"]]).strip()
    a_img_path = str(row[cols["after_image"]]).strip()
    b_msk_path = str(row[cols["before_mask"]]).strip()
    a_msk_path = str(row[cols["after_mask"]]).strip()

    emoji_print(f"\n🧩 Par #{idx_pair:04d}:")
    emoji_print(f"   🖼️ before: {b_img_path}")
    emoji_print(f"   🖼️ after : {a_img_path}")
    emoji_print(f"   🎭 before_mask: {b_msk_path}")
    emoji_print(f"   🎭 after_mask : {a_msk_path}")

    b_img = load_image(b_img_path)
    a_img = load_image(a_img_path)
    b_msk = load_image(b_msk_path)
    a_msk = load_image(a_msk_path)

    if any(x is None for x in [b_img, a_img, b_msk, a_msk]):
        emoji_print(
            f"⏩ Par #{idx_pair:04d}: saltado por archivos faltantes o ilegibles."
        )
        return False

    b_msk_g = to_grayscale_mask(b_msk)
    a_msk_g = to_grayscale_mask(a_msk)

    b_np = np.array(b_msk_g)
    a_np = np.array(a_msk_g)

    b_bbox = mask_to_bbox(b_np)
    a_bbox = mask_to_bbox(a_np)

    if b_bbox is None:
        emoji_print(
            f"⛔ Par #{idx_pair:04d}: máscara BEFORE vacía ➜ no se puede recortar."
        )
        return False
    if a_bbox is None:
        emoji_print(
            f"⛔ Par #{idx_pair:04d}: máscara AFTER vacía ➜ no se puede recortar."
        )
        return False

    b_bbox_sq = expand_square_bbox(
        b_bbox, b_img.width, b_img.height, margin_ratio=margin
    )
    a_bbox_sq = expand_square_bbox(
        a_bbox, a_img.width, a_img.height, margin_ratio=margin
    )

    b_img_c = crop_and_resize(b_img, b_bbox_sq, size=size)
    a_img_c = crop_and_resize(a_img, a_bbox_sq, size=size)
    b_msk_c = crop_and_resize(b_msk_g, b_bbox_sq, size=size)
    a_msk_c = crop_and_resize(a_msk_g, a_bbox_sq, size=size)

    # Guardar documento de texto con columnas extra
    txt_columns = [
        "before_split",
        "after_split",
        "before_image_abnormality_id",
        "after_image_abnormality_id",
        "breast_density",
        "mass_shape",
        "pathology",
        "view",
        "mass_margins",
        "assessment",
        "size_before_px",
        "size_after_px",
        "growth_pct",
    ]
    txt_path = os.path.join(pair_folder, "info.txt")
    with open(txt_path, "w") as f:
        for col in txt_columns:
            val = row.get(col, "")
            f.write(f"{col}: {val}\n")

    ext = (
        "png"
        if image_format.lower() == "png"
        else (
            "jpg" if image_format.lower() in ["jpg", "jpeg"] else image_format.lower()
        )
    )

    save_image(
        b_img_c,
        os.path.join(pair_folder, f"before_image.{ext}"),
        image_format,
        jpeg_quality,
    )
    save_image(
        a_img_c,
        os.path.join(pair_folder, f"after_image.{ext}"),
        image_format,
        jpeg_quality,
    )

    if b_msk.mode == "1":
        b_out = b_msk_c.convert("1")
    else:
        b_out = b_msk_c.convert("L")
    if a_msk.mode == "1":
        a_out = a_msk_c.convert("1")
    else:
        a_out = a_msk_c.convert("L")

    save_image(
        b_out, os.path.join(pair_folder, "before_image_mask.png"), "png", jpeg_quality
    )
    save_image(
        a_out, os.path.join(pair_folder, "after_image_mask.png"), "png", jpeg_quality
    )

    emoji_print(f"✅ Guardado par #{idx_pair:04d} en {pair_folder}")
    return True


def get_args():
    """
    SUMMARY
    -------
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Recorta pares de imágenes por la zona del tumor (máscara) y guarda 256x256 con la máxima calidad."
    )
    parser.add_argument(
        "--excel",
        default="../pairs.xlsx",
        help='Ruta al archivo Excel (primera hoja "pairs"). Por defecto "../pairs.xlsx".',
    )
    parser.add_argument(
        "--output-dir",
        default="../../../images",
        help='Directorio raíz donde se guardarán las carpetas de pares. Por defecto "../../images".',
    )
    parser.add_argument(
        "--size",
        type=int,
        default=256,
        help="Tamaño de salida (lado) en pixeles. Por defecto 256.",
    )
    parser.add_argument(
        "--image-format",
        default="png",
        choices=["png", "jpg", "jpeg", "tiff", "bmp"],
        help="Formato para guardar las imágenes (máscaras siempre en PNG).",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=100,
        help="Calidad JPEG (si se usa jpg/jpeg). Por defecto 100.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=1,
        help="Índice numérico inicial para nombrar carpetas (se mostrará como 0001, 0002, ...).",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Si se indica, no sobrescribe carpetas existentes con contenido.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Procesa solo los primeros N pares (por defecto procesa todos).",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.10,
        help="Margen extra alrededor de la caja de la máscara (proporción). Por defecto 0.10.",
    )
    parser.add_argument(
        "--col-before-image",
        default="before_image",
        help='Nombre de la columna para la imagen "before".',
    )
    parser.add_argument(
        "--col-after-image",
        default="after_image",
        help='Nombre de la columna para la imagen "after".',
    )
    parser.add_argument(
        "--col-before-mask",
        default="before_image_mask",
        help='Nombre de la columna para la máscara "before".',
    )
    parser.add_argument(
        "--col-after-mask",
        default="after_image_mask",
        help='Nombre de la columna para la máscara "after".',
    )

    return parser


def main():

    args = get_args()

    # excel_path = args.excel
    excel_path = "../pairs.xlsx"
    # out_dir = args.output_dir
    out_dir = "../../images"
    # overwrite = not args.no_overwrite
    overwrite = not True
    ensure_dir(out_dir)

    # Load Excel (always sheet "pairs")
    emoji_print("\n📖 Leyendo excel ...")
    try:
        df = pd.read_excel(excel_path, sheet_name="pairs")
    except Exception as e:
        emoji_print(f"💥 Error leyendo Excel: {excel_path}\n   ➜ {e}")
        sys.exit(1)

    col_map = {
        # "before_image": args.col_before_image,
        "before_image": "before_image",
        # "after_image": args.col_after_image,
        "after_image": "after_image",
        # "before_mask": args.col_before_mask,
        "before_mask": "before_image_mask",
        # "after_mask": args.col_after_mask,
        "after_mask": "after_image_mask",
    }

    try:
        validate_columns(
            df,
            [
                col_map["before_image"],
                col_map["after_image"],
                col_map["before_mask"],
                col_map["after_mask"],
            ],
        )
    except Exception as e:
        emoji_print(f"💥 {e}")
        sys.exit(2)

    total_rows = len(df)
    limit = None
    total = total_rows if limit is None else min(limit, total_rows)
    emoji_print(f"🧮 Filas a procesar: {total} (de {total_rows})")
    if limit is None:
        emoji_print("🔁 Sin límite: se procesarán todas las filas.")

    processed = 0
    skipped = 0

    # We'll iterate by dataframe index to respect start-index padding
    count = 0
    for _, row in df.iterrows():
        if limit is not None and count >= limit:
            break
        size = 256
        margin = 0.10
        jpeg_quality = 100
        image_format = "png"
        start_index = 1
        idx_pair = start_index + count
        ok = process_row(
            idx_pair=idx_pair,
            row=row,
            out_dir=out_dir,
            cols=col_map,
            size=size,
            image_format=image_format,
            jpeg_quality=jpeg_quality,
            overwrite=overwrite,
            margin=margin,
        )
        if ok:
            processed += 1
        else:
            skipped += 1

        count += 1
        if count % 10 == 0 or count == total:
            emoji_print(f"\n⏱️ Progreso: {count}/{total} pares")

    emoji_print(
        f"\n🎉 Listo. Procesados: {processed} | Saltados: {skipped} | Total leído: {total}"
    )


if __name__ == "__main__":
    main()
