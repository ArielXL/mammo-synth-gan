import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path


def save_chart(
    mammo_df: pd.DataFrame,
    dir_plot: Path,
    col_img: str,
    col_metric: str,
    name_img: str,
    title_chart: str,
    xlabel: str,
    ylabel: str,
    filename: str,
) -> None:
    """
    SUMMARY
    -------
    Guarda un gráfico de una métrica específica para una imagen dada.

    PARAMETERS
    ----------
    mammo_df : pd.DataFrame
        DataFrame que contiene los datos de mamografías.
    dir_plot : Path
        Directorio donde se guardará el gráfico.
    col_img : str
        Nombre de la columna que contiene los nombres de las imágenes.
    col_metric : str
        Nombre de la columna que contiene la métrica a graficar.
    name_img : str
        Nombre de la imagen a graficar.
    title_chart : str
        Título del gráfico.
    xlabel : str
        Etiqueta del eje x.
    ylabel : str
        Etiqueta del eje y.
    filename : str
        Nombre del archivo donde se guardará el gráfico.
    """
    df = mammo_df[mammo_df[col_img] == name_img]
    plt.figure(figsize=(10, 5))
    plt.plot(df[xlabel], df[col_metric], marker="o", linestyle="-")
    plt.xlabel("Subcarpetas")
    plt.ylabel(ylabel)
    plt.title(title_chart)

    n = len(df)
    if n > 0:
        idxs = np.array(list(range(0, n, 20)), dtype=int)
        if (n - 1) not in idxs:
            idxs = np.append(idxs, n - 1)
        xtick_labels = list(df[xlabel].iloc[idxs])
        plt.xticks(xtick_labels, rotation=45, ha="right")
    else:
        plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(os.path.join(dir_plot, filename))
    plt.close()


def save_chart_union(
    mammo_df: pd.DataFrame,
    dir_plot: Path,
    col_img: str,
    col_metric: str,
    name_img1: str,
    name_img2: str,
    title_chart: str,
    xlabel: str,
    ylabel: str,
    filename: str,
    label1: str,
    label2: str,
) -> None:
    """
    SUMMARY
    -------
    Guarda un gráfico que compara dos conjuntos de datos en un solo gráfico.

    PARAMETERS
    ----------
    mammo_df : pd.DataFrame
        DataFrame que contiene los datos de mamografías.
    dir_plot : Path
        Directorio donde se guardará el gráfico.
    col_img : str
        Nombre de la columna que contiene los nombres de las imágenes.
    col_metric : str
        Nombre de la columna que contiene la métrica a graficar.
    name_img1 : str
        Nombre de la primera imagen a graficar.
    name_img2 : str
        Nombre de la segunda imagen a graficar.
    title_chart : str
        Título del gráfico.
    xlabel : str
        Etiqueta del eje x.
    ylabel : str
        Etiqueta del eje y.
    filename : str
        Nombre del archivo donde se guardará el gráfico.
    label1 : str
        Etiqueta para la primera imagen en la leyenda.
    label2 : str
        Etiqueta para la segunda imagen en la leyenda.
    """

    df1 = mammo_df[mammo_df[col_img] == name_img1]
    df2 = mammo_df[mammo_df[col_img] == name_img2]
    plt.figure(figsize=(10, 5))
    plt.plot(
        df1[xlabel],
        df1[col_metric],
        marker="o",
        linestyle="-",
        label=label1 or name_img1,
    )
    plt.plot(
        df2[xlabel],
        df2[col_metric],
        marker="o",
        linestyle="-",
        label=label2 or name_img2,
    )
    plt.xlabel("Subcarpetas")
    plt.ylabel(ylabel)
    plt.title(title_chart)
    n1 = len(df1)
    n2 = len(df2)
    n = max(n1, n2)
    if n > 0:
        idxs = np.array(list(range(0, n, 20)), dtype=int)
        if n1 > 0 and (n1 - 1) not in idxs:
            idxs = np.append(idxs, n1 - 1)
        if n2 > 0 and (n2 - 1) not in idxs:
            idxs = np.append(idxs, n2 - 1)
        xtick_labels = set()
        if n1 > 0:
            xtick_labels.update(df1[xlabel].iloc[idxs[idxs < n1]])
        if n2 > 0:
            xtick_labels.update(df2[xlabel].iloc[idxs[idxs < n2]])
        xtick_labels = sorted(xtick_labels, key=lambda x: str(x))
        plt.xticks(xtick_labels, rotation=45, ha="right")
    else:
        plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(dir_plot, filename))
    plt.close()
