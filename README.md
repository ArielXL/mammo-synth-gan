# cGAN para Pronóstico de Evolución Tumoral Mamario

Este proyecto implementa un modelo cGAN (Pix2Pix) para el pronóstico de evolución tumoral usando pares de máscaras y síntesis de imagen mamográfica. Incluye métricas automáticas de calidad y exportación de resultados a CSV.

## Características

- Entrenamiento Pix2Pix cGAN con U-Net Generator y PatchGAN Discriminator.
- Etapa opcional 2: síntesis de imagen mamográfica a partir de la máscara generada.
- Evaluación automática: FID, SSIM, Dice, IoU, MSE, PSNR para imágenes y máscaras.
- Exportación de métricas: resultados guardados en CSV en la carpeta de salida.
- Checkpoints y reanudación automática.
- Gráficas de pérdidas y métricas.
- Configuración flexible por argumentos de línea de comandos.

## Estructura de carpetas

``` bash
cbis-ddsm-mass/
    mass_case_description_train_set.csv
    mass_case_description_test_set.csv
    Train/
    Test/
cgan/
    src/
    requirements.txt
images/
    0001/
    0002/
    ...
review-pairs/
    ...
```

## Ejemplo de uso

```bash
python3 main.py --gan-loss bce --ndf 128 --ngf 64 --lrD 1e-4 --lrG 5e-5 --lambda-L1 150 --lambda-L1-img 75 --lambda-SSIM 20 --lambda-TV 0.1 --label-smoothing --instance-noise 0.1 --instance-noise-end 0.01 --n-critic 1 --n-critic-strong 2 --alt-critic-period 10 --use-cosine --cosine-Tmax 80 --cosine-minlr-mult 0.05 --epochs 500 --batch-size 16 --enable-mammo-synthesis
```

## Argumentos principales

- `--gan-loss`: Tipo de pérdida adversarial (`bce` o `ls`).
- `--ndf`, `--ngf`: Filtros del discriminador/generador.
- `--lrD`, `--lrG`: Learning rates.
- `--lambda-L1`, `--lambda-L1-img`, `--lambda-SSIM`, `--lambda-TV`: Ponderación de pérdidas.
- `--epochs`, `--batch-size`: Épocas y tamaño de batch.
- `--enable-mammo-synthesis`: Activa la etapa de síntesis de imagen.

## Resultados

Al finalizar el entrenamiento, se generan:
- Imágenes y máscaras en `test_results/`
- Métricas de calidad en `metrics_results.csv`
- Gráficas en `plots/`
- Checkpoints en `checkpoints/`

## Requisitos

- Python 3.8+
- PyTorch
- torchmetrics
- pandas, numpy, matplotlib, PIL

Instala dependencias con:

```bash
pip install -r requirements.txt
```

## Contacto

Para dudas, mejoras o colaboración, contacta a [tu correo o github].
