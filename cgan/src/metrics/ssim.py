import numpy as np

from typing import List, Tuple

from PIL import Image

from skimage.metrics import structural_similarity as ssim


def load_gray_match_size(p1: str, p2: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    SUMMARY
    -------
    Carga dos imágenes, escala de grises, normaliza a [0,1] y ajusta tamaño de la 2 a la 1.

    PARAMETERS
    ----------
    p1 : str
        Ruta a la primera imagen.
    p2 : str
        Ruta a la segunda imagen.

    RETURNS
    -------
    Tuple[np.ndarray, np.ndarray]
        Dos arrays NumPy con las imágenes cargadas y procesadas.
    """

    # Permite modo de carga: 'L' (grises), '1' (binario)
    def open_img(path, mode):
        img = Image.open(path)
        if mode == "L":
            return img.convert("L")
        elif mode == "1":
            return img.convert("1")
        else:
            return img.convert("L")

    return open_img


def get_ssim(imgs1: List[str], imgs2: List[str], mode: str = "L") -> float:
    """
    SUMMARY
    -------
    Computa el índice de similitud estructural (SSIM) entre dos conjuntos de imágenes.

    PARAMETERS
    ----------
    imgs1 : List[str]
        Lista de rutas a las primeras imágenes.
    imgs2 : List[str]
        Lista de rutas a las segundas imágenes.

    RETURNS
    -------
    float
        El SSIM promedio entre las imágenes correspondientes de ambos conjuntos.

    """
    # Nuevo: modo de carga
    # if hasattr(get_ssim, "mode_override"):
    #     mode = getattr(get_ssim, "mode_override")
    n = min(len(imgs1), len(imgs2))
    scores = []
    for i in range(n):
        im1 = Image.open(imgs1[i]).convert(mode)
        im2 = Image.open(imgs2[i]).convert(mode)
        if im2.size != im1.size:
            im2 = im2.resize(im1.size, Image.BICUBIC)
        a = np.asarray(im1, dtype=np.float32) / 255.0
        b = np.asarray(im2, dtype=np.float32) / 255.0
        scores.append(ssim(a, b, data_range=1.0))
    return float(np.mean(scores)) if scores else float("nan")
