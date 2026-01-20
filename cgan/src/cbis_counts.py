import sys
import argparse
import pandas as pd

from pathlib import Path


TARGET_COLS = [
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
    "review",
]


def normalize(name: str) -> str:
    """
    SUMMARY
    -------
    Normaliza nombres de columnas: minúsculas, guiones/espacios -> "_".

    PARAMETERS
    ----------
    name : str
        Nombre de columna original.

    RETURNS
    -------
    str
        Nombre de columna normalizado.
    """
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")


def load_and_combine(excel_path: Path) -> pd.DataFrame:
    """
    SUMMARY
    -------
    Carga todas las hojas de un archivo Excel, normaliza los nombres de las columnas
    y devuelve un DataFrame combinado con solo las columnas de interés (si existen).

    PARAMETERS
    ----------
    excel_path : Path
        Ruta al archivo Excel.

    RETURNS
    -------
    pd.DataFrame
        DataFrame combinado con las columnas de interés.
    """
    xls = pd.ExcelFile(excel_path)
    frames = []
    for sheet in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet)
        df = df.rename(columns={c: normalize(c) for c in df.columns})
        cols_here = [c for c in TARGET_COLS if c in df.columns]
        if cols_here:
            frames.append(df[cols_here])
    if frames:
        return pd.concat(frames, ignore_index=True, sort=False)
    else:
        return pd.DataFrame(columns=TARGET_COLS)


def counts_for_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    SUMMARY
    -------
    Devuelve un DataFrame con los conteos de valores para una
    columna específica, incluyendo NaN. Si la columna no existe,
    devuelve un DataFrame indicando que no se encontró.

    PARAMETERS
    ----------
    df : pd.DataFrame
        DataFrame de entrada.
    col : str
        Nombre de la columna para la cual se desean los conteos.

    RETURNS
    -------
    pd.DataFrame
        DataFrame con dos columnas: la columna original y "count".
    """
    if col not in df.columns:
        return pd.DataFrame({col: ["<columna no encontrada>"], "count": [0]})
    vc = df[col].value_counts(dropna=False).reset_index()
    vc.columns = [col, "count"]
    vc[col] = vc[col].astype(object).where(~vc[col].isna(), "(NaN)")
    vc = vc.sort_values("count", ascending=False, kind="mergesort").reset_index(
        drop=True
    )
    return vc


def print_section(title: str, char: str = "=") -> None:
    """
    SUMMARY
    -------
    Imprime un título de sección con un subrayado.

    PARAMETERS
    ----------
    title : str
        Título de la sección.
    char : str, optional
        Carácter para el subrayado (por defecto es '=').
    """
    print("\n" + title)
    print(char * len(title))


def main():
    parser = argparse.ArgumentParser(
        description="Imprime conteos por categoría de columnas CBIS-DDSM."
    )
    parser.add_argument(
        "--excel",
        type=Path,
        default="../pairs.xlsx",
        help="Ruta al archivo .xlsx de entrada.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=50,
        help="Máximo de filas a imprimir por columna (usar -1 para imprimir todas).",
    )
    args = parser.parse_args()

    if not args.excel.exists():
        print(f"ERROR: no se encuentra el archivo: {args.excel}", file=sys.stderr)
        sys.exit(1)

    pd.set_option("display.max_rows", None if args.top == -1 else args.top)
    pd.set_option("display.max_colwidth", 200)
    pd.set_option("display.width", 200)
    pd.set_option("display.colheader_justify", "left")

    df = load_and_combine(args.excel)
    total = len(df)
    print_section("Resumen general")
    print(f"Filas totales combinadas (todas las hojas): {total}")
    print(f"Columnas objetivo: {', '.join(TARGET_COLS)}")

    overview_rows = []
    counts_tables = {}

    for col in TARGET_COLS:
        if col in df.columns:
            non_null = int(df[col].notna().sum())
            nulls = total - non_null
            vc = counts_for_column(df, col)
            n_cats = vc.shape[0]
            overview_rows.append(
                {
                    "columna": col,
                    "presente_en_excel": True,
                    "filas_totales": total,
                    "no_nulos": non_null,
                    "nulos": nulls,
                    "categorias_unicas": int(n_cats),
                }
            )
            counts_tables[col] = vc
        else:
            overview_rows.append(
                {
                    "columna": col,
                    "presente_en_excel": False,
                    "filas_totales": total,
                    "no_nulos": 0,
                    "nulos": total,
                    "categorias_unicas": 0,
                }
            )

    overview_df = pd.DataFrame(overview_rows)
    print_section("Overview de columnas")
    print(overview_df.to_string(index=False))

    for col in TARGET_COLS:
        print_section(f"Conteos para: {col}", "-")
        if col in counts_tables:
            vc = counts_tables[col]
            if args.top != -1:
                vc_to_print = vc.head(args.top)
            else:
                vc_to_print = vc
            print(vc_to_print.to_string(index=False))
        else:
            print("<columna no encontrada en el Excel>")


if __name__ == "__main__":
    main()
