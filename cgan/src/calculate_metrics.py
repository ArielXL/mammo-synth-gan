import sys
import pandas as pd

from pathlib import Path


def compute_means(excel_path: str):
    """
    Calcula el promedio de SSIM y LPIPS para las filas donde:
    img1 == 'after_image.png' y img2 == 'generated_mammo.png'
    """

    excel_path = Path(excel_path)

    if not excel_path.exists():
        raise FileNotFoundError(f"No existe el archivo: {excel_path}")

    # Leer Excel
    df = pd.read_excel(excel_path)

    required_cols = {"img1", "img2", "SSIM", "LPIPS"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"Faltan columnas en el Excel: {missing}")

    # Filtrado
    filtered_after_generated = df[
        (df["img1"] == "after_image.png") & (df["img2"] == "generated_mammo.png")
    ]
    filtered_before_generated = df[
        (df["img1"] == "before_image.png") & (df["img2"] == "generated_mammo.png")
    ]

    if filtered_after_generated.empty:
        raise ValueError("No se encontraron filas para 'after_image.png'.")
    if filtered_before_generated.empty:
        raise ValueError("No se encontraron filas para 'before_image.png'.")

    # Conversión segura a numérico
    filtered_after_generated["SSIM"] = pd.to_numeric(
        filtered_after_generated["SSIM"], errors="coerce"
    )
    filtered_after_generated["LPIPS"] = pd.to_numeric(
        filtered_after_generated["LPIPS"], errors="coerce"
    )
    filtered_before_generated["SSIM"] = pd.to_numeric(
        filtered_before_generated["SSIM"], errors="coerce"
    )
    filtered_before_generated["LPIPS"] = pd.to_numeric(
        filtered_before_generated["LPIPS"], errors="coerce"
    )

    mean_ssim_after_generated = filtered_after_generated["SSIM"].mean()
    mean_lpips_after_generated = filtered_after_generated["LPIPS"].mean()
    mean_ssim_before_generated = filtered_before_generated["SSIM"].mean()
    mean_lpips_before_generated = filtered_before_generated["LPIPS"].mean()

    return (
        mean_ssim_after_generated,
        mean_lpips_after_generated,
        len(filtered_after_generated),
        mean_ssim_before_generated,
        mean_lpips_before_generated,
        len(filtered_before_generated),
    )


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Uso:")
        print("  python compute_metrics.py ruta/al/archivo.xlsx")
        sys.exit(1)

    # python3 compute_metrics.py ruta/al/archivo.xlsx
    excel_file = sys.argv[1]

    try:
        (
            mean_ssim_after,
            mean_lpips_after,
            n_after,
            mean_ssim_before,
            mean_lpips_before,
            n_before,
        ) = compute_means(excel_file)

        print("RESULTADOS")
        print("==========")
        print(f"Número de filas consideradas (after) : {n_after}")
        print(f"Promedio SSIM (after)                : {mean_ssim_after:.6f}")
        print(f"Promedio LPIPS (after)               : {mean_lpips_after:.6f}")
        print(f"Número de filas consideradas (before): {n_before}")
        print(f"Promedio SSIM (before)               : {mean_ssim_before:.6f}")
        print(f"Promedio LPIPS (before)              : {mean_lpips_before:.6f}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
