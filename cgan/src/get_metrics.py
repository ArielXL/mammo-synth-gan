import os
import torch
import lpips
import argparse
import numpy as np
import pandas as pd

from pathlib import Path

from typing import List

from PIL import Image

from metrics.ssim import get_ssim
from metrics.chart import save_chart, save_chart_union


def compute_lpips(
    img1_path: str, img2_path: str, net: str = "alex", mode: str = "L"
) -> float:
    """
    SUMMARY
    -------
    Calcula LPIPS entre dos imágenes usando la red especificada (alex, vgg, squeeze).

    PARAMETERS
    ----------
    img1_path : str
        Ruta a la primera imagen.
    img2_path : str
        Ruta a la segunda imagen.
    net : str, optional
        Red a usar para LPIPS (default es 'alex').
    mode : str, optional
        Modo de apertura de imagen ('RGB', 'L', '1').

    RETURNS
    -------
    float
        Valor LPIPS calculado.
    """
    loss_fn = lpips.LPIPS(net=net)

    def img_to_lpips_tensor(path, mode="L"):
        img = Image.open(path).convert(mode)
        img = np.array(img).astype(np.float32) / 255.0
        if mode == "RGB":
            img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0) * 2 - 1
        else:
            img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0) * 2 - 1
        return img

    img1 = img_to_lpips_tensor(img1_path, mode)
    img2 = img_to_lpips_tensor(img2_path, mode)
    with torch.no_grad():
        lpips_val = loss_fn(img1, img2).item()
    return lpips_val


def compute_metrics(img1_path: str, img2_path: str) -> float:
    """
    SUMMARY
    -------
    Computa FID, KID y SSIM entre dos imágenes dadas por sus rutas.

    PARAMETERS
    ----------
    img1_path : str
        Ruta a la primera imagen.
    img2_path : str
        Ruta a la segunda imagen.

    RETURNS
    -------
    Tuple[float, float]
        KID y SSIM calculados.
    """
    imgs1 = [img1_path]
    imgs2 = [img2_path]

    ssim_val = get_ssim(imgs1, imgs2)

    return ssim_val


def iter_subdirs(root: Path) -> List[Path]:
    """
    SUMMARY
    -------
    Encuentra todas las subcarpetas bajo 'root', excluyendo 'root' mismo.

    PARAMETERS
    ----------
    root : Path
        Carpeta raíz donde buscar subcarpetas.

    RETURNS
    -------
    List[Path]
        Lista de subcarpetas encontradas.
    """
    subdirs = set()
    for dirpath, _, _ in os.walk(root):
        p = Path(dirpath)
        if p == root:
            continue
        subdirs.add(p)
    return sorted(subdirs)


def ssim_charts(
    mammo_df: pd.DataFrame,
    mask_df: pd.DataFrame,
    dir_plot: Path,
    col_img: str,
    col_metric_ssim: str,
    after_image: str,
    before_image: str,
    after_image_mask: str,
    before_image_mask: str,
    title_after_img_mammo_chart_ssim: str,
    title_before_img_mammo_chart_ssim: str,
    title_after_mask_mammo_chart_ssim: str,
    title_before_mask_mammo_chart_ssim: str,
    title_image_union_chart_ssim: str,
    title_mask_union_chart_ssim: str,
    xlabel: str,
    ylabel_after_img_mammo_ssim: str,
    ylabel_before_img_mammo_ssim: str,
    ylabel_after_mask_mammo_ssim: str,
    ylabel_before_mask_mammo_ssim: str,
    ylabel_image_union_ssim: str,
    ylabel_mask_union_ssim: str,
    filename_after_img_mammo_ssim: str,
    filename_before_img_mammo_ssim: str,
    filename_after_mask_mammo_ssim: str,
    filename_before_mask_mammo_ssim: str,
    filename_image_union_chart_ssim: str,
    filename_mask_union_chart_ssim: str,
):
    save_chart(
        mammo_df=mammo_df,
        dir_plot=dir_plot,
        col_img=col_img,
        col_metric=col_metric_ssim,
        name_img=after_image,
        title_chart=title_after_img_mammo_chart_ssim,
        xlabel=xlabel,
        ylabel=ylabel_after_img_mammo_ssim,
        filename=filename_after_img_mammo_ssim,
    )
    save_chart(
        mammo_df=mammo_df,
        dir_plot=dir_plot,
        col_img=col_img,
        col_metric=col_metric_ssim,
        name_img=before_image,
        title_chart=title_before_img_mammo_chart_ssim,
        xlabel=xlabel,
        ylabel=ylabel_before_img_mammo_ssim,
        filename=filename_before_img_mammo_ssim,
    )
    save_chart(
        mammo_df=mask_df,
        dir_plot=dir_plot,
        col_img=col_img,
        col_metric=col_metric_ssim,
        name_img=after_image_mask,
        title_chart=title_after_mask_mammo_chart_ssim,
        xlabel=xlabel,
        ylabel=ylabel_after_mask_mammo_ssim,
        filename=filename_after_mask_mammo_ssim,
    )
    save_chart(
        mammo_df=mask_df,
        dir_plot=dir_plot,
        col_img=col_img,
        col_metric=col_metric_ssim,
        name_img=before_image_mask,
        title_chart=title_before_mask_mammo_chart_ssim,
        xlabel=xlabel,
        ylabel=ylabel_before_mask_mammo_ssim,
        filename=filename_before_mask_mammo_ssim,
    )
    save_chart_union(
        mammo_df=mammo_df,
        dir_plot=dir_plot,
        col_img=col_img,
        col_metric=col_metric_ssim,
        name_img1=after_image,
        name_img2=before_image,
        title_chart=title_image_union_chart_ssim,
        xlabel=xlabel,
        ylabel=ylabel_image_union_ssim,
        filename=filename_image_union_chart_ssim,
        label1=after_image,
        label2=before_image,
    )
    save_chart_union(
        mammo_df=mask_df,
        dir_plot=dir_plot,
        col_img=col_img,
        col_metric=col_metric_ssim,
        name_img1=after_image_mask,
        name_img2=before_image_mask,
        title_chart=title_mask_union_chart_ssim,
        xlabel=xlabel,
        ylabel=ylabel_mask_union_ssim,
        filename=filename_mask_union_chart_ssim,
        label1=after_image_mask,
        label2=before_image_mask,
    )


def lpips_charts(
    mammo_df: pd.DataFrame,
    mask_df: pd.DataFrame,
    dir_plot: Path,
    col_img: str,
    col_metric_lpips: str,
    after_image: str,
    before_image: str,
    after_image_mask: str,
    before_image_mask: str,
    title_after_img_mammo_chart_lpips: str,
    title_before_img_mammo_chart_lpips: str,
    title_after_mask_mammo_chart_lpips: str,
    title_before_mask_mammo_chart_lpips: str,
    title_image_union_chart_lpips: str,
    title_mask_union_chart_lpips: str,
    xlabel: str,
    ylabel_after_img_mammo_lpips: str,
    ylabel_before_img_mammo_lpips: str,
    ylabel_after_mask_mammo_lpips: str,
    ylabel_before_mask_mammo_lpips: str,
    ylabel_image_union_lpips: str,
    ylabel_mask_union_lpips: str,
    filename_after_img_mammo_lpips: str,
    filename_before_img_mammo_lpips: str,
    filename_after_mask_mammo_lpips: str,
    filename_before_mask_mammo_lpips: str,
    filename_image_union_chart_lpips: str,
    filename_mask_union_chart_lpips: str,
):
    save_chart(
        mammo_df=mammo_df,
        dir_plot=dir_plot,
        col_img=col_img,
        col_metric=col_metric_lpips,
        name_img=after_image,
        title_chart=title_after_img_mammo_chart_lpips,
        xlabel=xlabel,
        ylabel=ylabel_after_img_mammo_lpips,
        filename=filename_after_img_mammo_lpips,
    )
    save_chart(
        mammo_df=mammo_df,
        dir_plot=dir_plot,
        col_img=col_img,
        col_metric=col_metric_lpips,
        name_img=before_image,
        title_chart=title_before_img_mammo_chart_lpips,
        xlabel=xlabel,
        ylabel=ylabel_before_img_mammo_lpips,
        filename=filename_before_img_mammo_lpips,
    )
    save_chart(
        mammo_df=mask_df,
        dir_plot=dir_plot,
        col_img=col_img,
        col_metric=col_metric_lpips,
        name_img=after_image_mask,
        title_chart=title_after_mask_mammo_chart_lpips,
        xlabel=xlabel,
        ylabel=ylabel_after_mask_mammo_lpips,
        filename=filename_after_mask_mammo_lpips,
    )
    save_chart(
        mammo_df=mask_df,
        dir_plot=dir_plot,
        col_img=col_img,
        col_metric=col_metric_lpips,
        name_img=before_image_mask,
        title_chart=title_before_mask_mammo_chart_lpips,
        xlabel=xlabel,
        ylabel=ylabel_before_mask_mammo_lpips,
        filename=filename_before_mask_mammo_lpips,
    )
    save_chart_union(
        mammo_df=mammo_df,
        dir_plot=dir_plot,
        col_img=col_img,
        col_metric=col_metric_lpips,
        name_img1=after_image,
        name_img2=before_image,
        title_chart=title_image_union_chart_lpips,
        xlabel=xlabel,
        ylabel=ylabel_image_union_lpips,
        filename=filename_image_union_chart_lpips,
        label1=after_image,
        label2=before_image,
    )
    save_chart_union(
        mammo_df=mask_df,
        dir_plot=dir_plot,
        col_img=col_img,
        col_metric=col_metric_lpips,
        name_img1=after_image_mask,
        name_img2=before_image_mask,
        title_chart=title_mask_union_chart_lpips,
        xlabel=xlabel,
        ylabel=ylabel_mask_union_lpips,
        filename=filename_mask_union_chart_lpips,
        label1=after_image_mask,
        label2=before_image_mask,
    )


def save_charts(
    mammo_df: pd.DataFrame,
    mask_df: pd.DataFrame,
    dir_plot: Path,
    col_img: str,
    col_metric_ssim: str,
    col_metric_lpips: str,
    after_image: str,
    before_image: str,
    after_image_mask: str,
    before_image_mask: str,
    title_after_img_mammo_chart_ssim: str,
    title_before_img_mammo_chart_ssim: str,
    title_after_mask_mammo_chart_ssim: str,
    title_before_mask_mammo_chart_ssim: str,
    title_image_union_chart_ssim: str,
    title_mask_union_chart_ssim: str,
    title_after_img_mammo_chart_lpips: str,
    title_before_img_mammo_chart_lpips: str,
    title_after_mask_mammo_chart_lpips: str,
    title_before_mask_mammo_chart_lpips: str,
    title_image_union_chart_lpips: str,
    title_mask_union_chart_lpips: str,
    xlabel: str,
    ylabel_after_img_mammo_ssim: str,
    ylabel_before_img_mammo_ssim: str,
    ylabel_after_mask_mammo_ssim: str,
    ylabel_before_mask_mammo_ssim: str,
    ylabel_image_union_ssim: str,
    ylabel_mask_union_ssim: str,
    filename_after_img_mammo_ssim: str,
    filename_before_img_mammo_ssim: str,
    filename_after_mask_mammo_ssim: str,
    filename_before_mask_mammo_ssim: str,
    filename_image_union_chart_ssim: str,
    filename_mask_union_chart_ssim: str,
    ylabel_after_img_mammo_lpips: str,
    ylabel_before_img_mammo_lpips: str,
    ylabel_after_mask_mammo_lpips: str,
    ylabel_before_mask_mammo_lpips: str,
    ylabel_image_union_lpips: str,
    ylabel_mask_union_lpips: str,
    filename_after_img_mammo_lpips: str,
    filename_before_img_mammo_lpips: str,
    filename_after_mask_mammo_lpips: str,
    filename_before_mask_mammo_lpips: str,
    filename_image_union_chart_lpips: str,
    filename_mask_union_chart_lpips: str,
) -> None:
    ssim_charts(
        mammo_df,
        mask_df,
        dir_plot,
        col_img,
        col_metric_ssim,
        after_image,
        before_image,
        after_image_mask,
        before_image_mask,
        title_after_img_mammo_chart_ssim,
        title_before_img_mammo_chart_ssim,
        title_after_mask_mammo_chart_ssim,
        title_before_mask_mammo_chart_ssim,
        title_image_union_chart_ssim,
        title_mask_union_chart_ssim,
        xlabel,
        ylabel_after_img_mammo_ssim,
        ylabel_before_img_mammo_ssim,
        ylabel_after_mask_mammo_ssim,
        ylabel_before_mask_mammo_ssim,
        ylabel_image_union_ssim,
        ylabel_mask_union_ssim,
        filename_after_img_mammo_ssim,
        filename_before_img_mammo_ssim,
        filename_after_mask_mammo_ssim,
        filename_before_mask_mammo_ssim,
        filename_image_union_chart_ssim,
        filename_mask_union_chart_ssim,
    )
    lpips_charts(
        mammo_df,
        mask_df,
        dir_plot,
        col_img,
        col_metric_lpips,
        after_image,
        before_image,
        after_image_mask,
        before_image_mask,
        title_after_img_mammo_chart_lpips,
        title_before_img_mammo_chart_lpips,
        title_after_mask_mammo_chart_lpips,
        title_before_mask_mammo_chart_lpips,
        title_image_union_chart_lpips,
        title_mask_union_chart_lpips,
        xlabel,
        ylabel_after_img_mammo_lpips,
        ylabel_before_img_mammo_lpips,
        ylabel_after_mask_mammo_lpips,
        ylabel_before_mask_mammo_lpips,
        ylabel_image_union_lpips,
        ylabel_mask_union_lpips,
        filename_after_img_mammo_lpips,
        filename_before_img_mammo_lpips,
        filename_after_mask_mammo_lpips,
        filename_before_mask_mammo_lpips,
        filename_image_union_chart_lpips,
        filename_mask_union_chart_lpips,
    )


def get_args():
    parser = argparse.ArgumentParser(
        description="Recorre subcarpetas y calcula SSIM y LPIPS para dos pares de imágenes por subcarpeta."
    )
    parser.add_argument(
        "--root",
        default="../runs/test_results/",
        help="Carpeta raíz que contiene subcarpetas a evaluar",
    )
    parser.add_argument(
        "--out",
        default="../runs/metrics/metrics_results.xlsx",
        help="Ruta del archivo Excel de salida",
    )
    parser.add_argument(
        "--plot",
        default="../runs/metrics/",
        help="Directorio donde guardar los gráficos",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Peso para la combinación de métricas (default 0.5)",
    )
    return parser.parse_args()


def main():
    args = get_args()
    os.makedirs(args.plot, exist_ok=True)

    root = Path(args.root)
    if not root.is_dir():
        raise SystemExit(f"[ERROR] La ruta raíz no es una carpeta: {root}")

    rows = []
    subdirs = iter_subdirs(root)
    if not subdirs:
        print(
            "[AVISO] No se encontraron subcarpetas bajo la raíz. ¿Seguro que la estructura es correcta?"
        )

    pairs = [
        ("after_image_mask.png", "generated_mask.png", "mask_pair"),
        ("after_image.png", "generated_mammo.png", "mammo_pair"),
        ("before_image_mask.png", "generated_mask.png", "mask_pair"),
        ("before_image.png", "generated_mammo.png", "mammo_pair"),
    ]

    alpha = args.alpha
    for subdir in subdirs:
        before_img = subdir / "before_image.png"
        after_img = subdir / "after_image.png"
        generated_img = subdir / "generated_mammo.png"
        before_mask = subdir / "before_image_mask.png"
        after_mask = subdir / "after_image_mask.png"
        generated_mask = subdir / "generated_mask.png"

        # mammo_pair: escala de grises
        ssim_before = get_ssim([str(before_img)], [str(generated_img)], mode="L")
        lpips_before = compute_lpips(
            str(before_img), str(generated_img), net="alex", mode="L"
        )
        combined_score = alpha * (1 - ssim_before) + (1 - alpha) * lpips_before

        # mask_pair: blanco y negro binario
        ssim_before_mask = get_ssim([str(before_mask)], [str(generated_mask)], mode="1")
        lpips_before_mask = compute_lpips(
            str(before_mask), str(generated_mask), net="alex", mode="1"
        )
        combined_score_mask = (
            alpha * (1 - ssim_before_mask) + (1 - alpha) * lpips_before_mask
        )

        for a_name, b_name, pair_tag in pairs:
            a_path = subdir / a_name
            b_path = subdir / b_name
            ssim, lpips_val = 0.0, 0.0
            if pair_tag == "mammo_pair":
                ssim = compute_metrics(str(a_path), str(b_path))
                lpips_val = compute_lpips(
                    str(a_path), str(b_path), net="alex", mode="L"
                )
            elif pair_tag == "mask_pair":
                ssim = compute_metrics(str(a_path), str(b_path))
                lpips_val = compute_lpips(
                    str(a_path), str(b_path), net="alex", mode="1"
                )
            row = {
                "subfolder": str(subdir.relative_to(root)),
                "pair": pair_tag,
                "img1": a_name,
                "img2": b_name,
                "SSIM": ssim,
                "LPIPS": lpips_val,
            }
            if pair_tag == "mammo_pair":
                row["combined_score"] = combined_score
            elif pair_tag == "mask_pair":
                row["combined_score"] = combined_score_mask
            rows.append(row)
            print(f"✔ {subdir} | {pair_tag}: SSIM={ssim:.6f}, LPIPS={lpips_val:.6f}")

    df = pd.DataFrame(rows)
    mammo_df = df[df["pair"] == "mammo_pair"]
    mask_df = df[df["pair"] == "mask_pair"]

    save_charts(
        mammo_df=mammo_df,
        mask_df=mask_df,
        dir_plot=args.plot,
        col_img="img1",
        col_metric_ssim="SSIM",
        col_metric_lpips="LPIPS",
        after_image="after_image.png",
        before_image="before_image.png",
        after_image_mask="after_image_mask.png",
        before_image_mask="before_image_mask.png",
        title_after_img_mammo_chart_ssim="SSIM: after_image vs generated_mammo por subcarpeta",
        title_before_img_mammo_chart_ssim="SSIM: before_image vs generated_mammo por subcarpeta",
        title_after_mask_mammo_chart_ssim="SSIM: after_mask vs generated_mask por subcarpeta",
        title_before_mask_mammo_chart_ssim="SSIM: before_mask vs generated_mask por subcarpeta",
        title_image_union_chart_ssim="SSIM: after/before_image vs generated_mammo por subcarpeta",
        title_mask_union_chart_ssim="SSIM: after/before_mask vs generated_mask por subcarpeta",
        title_after_img_mammo_chart_lpips="LPIPS: after_image vs generated_mammo por subcarpeta",
        title_before_img_mammo_chart_lpips="LPIPS: before_image vs generated_mammo por subcarpeta",
        title_after_mask_mammo_chart_lpips="LPIPS: after_mask vs generated_mask por subcarpeta",
        title_before_mask_mammo_chart_lpips="LPIPS: before_mask vs generated_mask por subcarpeta",
        title_image_union_chart_lpips="LPIPS: after/before_image vs generated_mammo por subcarpeta",
        title_mask_union_chart_lpips="LPIPS: after/before_mask vs generated_mask por subcarpeta",
        xlabel="subfolder",
        ylabel_after_img_mammo_ssim="SSIM (after_image, generated_mammo)",
        ylabel_before_img_mammo_ssim="SSIM (before_image, generated_mammo)",
        ylabel_after_mask_mammo_ssim="SSIM (after_mask, generated_mask)",
        ylabel_before_mask_mammo_ssim="SSIM (before_mask, generated_mask)",
        ylabel_image_union_ssim="SSIM (image, generated_mammo)",
        ylabel_mask_union_ssim="SSIM (mask, generated_mask)",
        filename_after_img_mammo_ssim="ssim_after_img_vs_generated_mammo.png",
        filename_before_img_mammo_ssim="ssim_before_img_vs_generated_mammo.png",
        filename_after_mask_mammo_ssim="ssim_after_mask_vs_generated_mask.png",
        filename_before_mask_mammo_ssim="ssim_before_mask_vs_generated_mask.png",
        filename_image_union_chart_ssim="ssim_union_img_vs_generated_mammo.png",
        filename_mask_union_chart_ssim="ssim_union_mask_vs_generated_mask.png",
        ylabel_after_img_mammo_lpips="LPIPS (after_image, generated_mammo)",
        ylabel_before_img_mammo_lpips="LPIPS (before_image, generated_mammo)",
        ylabel_after_mask_mammo_lpips="LPIPS (after_mask, generated_mask)",
        ylabel_before_mask_mammo_lpips="LPIPS (before_mask, generated_mask)",
        ylabel_image_union_lpips="LPIPS (image, generated_mammo)",
        ylabel_mask_union_lpips="LPIPS (mask, generated_mask)",
        filename_after_img_mammo_lpips="lpips_after_img_vs_generated_mammo.png",
        filename_before_img_mammo_lpips="lpips_before_img_vs_generated_mammo.png",
        filename_after_mask_mammo_lpips="lpips_after_mask_vs_generated_mask.png",
        filename_before_mask_mammo_lpips="lpips_before_mask_vs_generated_mask.png",
        filename_image_union_chart_lpips="lpips_union_img_vs_generated_mammo.png",
        filename_mask_union_chart_lpips="lpips_union_mask_vs_generated_mask.png",
    )

    with pd.ExcelWriter(args.out, engine="openpyxl") as writer:
        mammo_df.to_excel(writer, sheet_name="mammo_pair", index=False)
        mask_df.to_excel(writer, sheet_name="mask_pair", index=False)

    print(f"\n✅ Listo. Resultados guardados en: {args.out}")


if __name__ == "__main__":
    main()
