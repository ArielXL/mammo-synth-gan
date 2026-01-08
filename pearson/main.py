import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import pearsonr


def downsample_idx(vec, cap):
    vec = np.asarray(vec)
    rng = np.random.default_rng(0)
    if vec.size <= cap:
        return vec
    return rng.choice(vec, size=cap, replace=False)


def main(dir, count, save_dir=None):
    excel_path = dir + "metrics_results.xlsx"

    df = pd.read_excel(
        excel_path,
        sheet_name=0,
        header=0,
    )

    id_col_name = "subfolder"
    ssim_col_name = "SSIM"
    lpips_col_name = "LPIPS"

    N_rows = df.shape[0]
    N = N_rows // 2

    odd_idx = 2 * np.arange(N)
    even_idx = 2 * np.arange(N) + 1

    case_numbers = df.iloc[odd_idx][id_col_name].astype(str).to_numpy()

    SSIM_all = pd.to_numeric(df[ssim_col_name], errors="coerce").to_numpy()
    LPIPS_all = pd.to_numeric(df[lpips_col_name], errors="coerce").to_numpy()

    SSIM_AG = 1.0 - SSIM_all[odd_idx]
    SSIM_BG = 1.0 - SSIM_all[even_idx]
    DS = SSIM_AG - SSIM_BG

    LPIPS_AG = LPIPS_all[odd_idx]
    LPIPS_BG = LPIPS_all[even_idx]
    DL = LPIPS_AG - LPIPS_BG

    print("DL NaNs:", np.isnan(DL).sum(), "/", DL.size)
    print("DS NaNs:", np.isnan(DS).sum(), "/", DS.size)

    caps = {
        "best": 12,
        "worst": 12,
        "ambiguous": 20,
        "conflict1": 12,
        "conflict2": 12,
    }

    ambig_pct = 25.0
    pos_pct = 65.0

    tauL = np.nanpercentile(DL, pos_pct)
    tauS = np.nanpercentile(DS, pos_pct)
    neg_tauL = np.nanpercentile(DL, 100.0 - pos_pct)
    neg_tauS = np.nanpercentile(DS, 100.0 - pos_pct)

    dist = np.sqrt(DL**2 + DS**2)
    rAmbig = np.nanpercentile(dist, ambig_pct)
    inAmbig = np.where(dist <= rAmbig)[0]
    ambiguous_idx = inAmbig.copy()

    isBest = (DL > tauL) & (DS > tauS)
    best_idx = np.setdiff1d(np.where(isBest)[0], inAmbig)

    isWorst = (DL < neg_tauL) & (DS < neg_tauS)
    worst_idx = np.setdiff1d(np.where(isWorst)[0], inAmbig)

    isConflict1 = (DL > tauL) & (DS < neg_tauS)
    conflict1_idx = np.setdiff1d(np.where(isConflict1)[0], inAmbig)

    isConflict2 = (DL < neg_tauL) & (DS > tauS)
    conflict2_idx = np.setdiff1d(np.where(isConflict2)[0], inAmbig)

    best_idx = downsample_idx(best_idx, caps["best"])
    worst_idx = downsample_idx(worst_idx, caps["worst"])
    ambiguous_idx = downsample_idx(ambiguous_idx, caps["ambiguous"])
    conflict1_idx = downsample_idx(conflict1_idx, caps["conflict1"])
    conflict2_idx = downsample_idx(conflict2_idx, caps["conflict2"])

    selected_all = np.unique(
        np.concatenate(
            [best_idx, worst_idx, ambiguous_idx, conflict1_idx, conflict2_idx]
        )
    )

    best_idx = best_idx.astype(int).ravel()
    worst_idx = worst_idx.astype(int).ravel()
    ambiguous_idx = ambiguous_idx.astype(int).ravel()
    conflict1_idx = conflict1_idx.astype(int).ravel()
    conflict2_idx = conflict2_idx.astype(int).ravel()
    selected_all = selected_all.astype(int).ravel()

    selection = {
        "best": best_idx,
        "worst": worst_idx,
        "ambiguous": ambiguous_idx,
        "conflict1": conflict1_idx,
        "conflict2": conflict2_idx,
        "all": selected_all,
    }

    print("\nFINAL SELECTION SUMMARY:")
    print(f"  N total cases:                {N}")
    print(
        f"  Ambiguous radius (rAmbig):    {rAmbig:.4g} "
        f"({ambig_pct:.1f} percentile of distances)"
    )
    print(f"  Threshold tau_LPIPS (pos pct):  {tauL:.4g} " f"({pos_pct:.0f} pctl)")
    print(f"  Threshold tau_SSIM (pos pct): {tauS:.4g} " f"({pos_pct:.0f} pctl)")
    print("  -------------------------------------------")
    print(f"  Best predictions:             {best_idx.size:3d}")
    print(f"  Worst predictions:            {worst_idx.size:3d}")
    print(f"  Ambiguous:                    {ambiguous_idx.size:3d}")
    print(f"  Conflict type 1:              {conflict1_idx.size:3d}")
    print(f"  Conflict type 2:              {conflict2_idx.size:3d}")
    print("  -------------------------------------------")
    print(f"  Total unique selected:        {selected_all.size:3d} / {N}\n")

    valid_mask = ~np.isnan(DL) & ~np.isnan(DS)
    pearson_coef, pearson_p = pearsonr(DL[valid_mask], DS[valid_mask])
    print(f"Coeficiente de Pearson DL vs DS: {pearson_coef:.4f} (p={pearson_p:.4g})\n")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title(r"$Espacio \ de \ contraste : LPIPS \ vs \ SSIM$")

    ax.scatter(DL, DS, s=20, color="0.6", label="Todos los casos")

    if best_idx.size:
        ax.scatter(DL[best_idx], DS[best_idx], s=80, label="G es más como B")
    if worst_idx.size:
        ax.scatter(DL[worst_idx], DS[worst_idx], s=80, label="G es más como A")
    if ambiguous_idx.size:
        ax.scatter(DL[ambiguous_idx], DS[ambiguous_idx], s=80, label="Ambigüos")
    if conflict1_idx.size:
        ax.scatter(
            DL[conflict1_idx], DS[conflict1_idx], s=80, marker="p", label="Conflicto 1"
        )
    if conflict2_idx.size:
        ax.scatter(
            DL[conflict2_idx], DS[conflict2_idx], s=80, marker="s", label="Conflicto 2"
        )

    theta = np.linspace(0, 2 * np.pi, 400)
    xC = rAmbig * np.cos(theta)
    yC = rAmbig * np.sin(theta)
    ax.plot(xC, yC, "k--", linewidth=1.2, label="Círculo de ambigüedad")

    ax.set_xlabel(r"$D_{LPIPS} = LPIPS_{antes-generada} - LPIPS_{después-generada}$")
    ax.set_ylabel(r"$D_{SSIM} = SSIM_{antes-generada} - SSIM_{después-generada}$")
    ax.set_aspect("equal", "box")
    ax.grid(True)
    ax.legend(loc="best")
    fig.tight_layout()

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"exp{count}_pearson.png")
        plt.savefig(save_path)
        print(f"Gráfica guardada en: {save_path}")
    plt.show()

    return selection, case_numbers, DL, DS, pearson_coef


if __name__ == "__main__":
    dirs = [
        "/home/ariel/Downloads/exp1_metrics/",
        "/home/ariel/Downloads/exp2_metrics/",
        "/home/ariel/Downloads/exp3_metrics/",
        "/home/ariel/Downloads/exp4_metrics/",
        "/home/ariel/Downloads/exp5_metrics/",
        "/home/ariel/Downloads/exp6_metrics/",
        "/home/ariel/Downloads/exp7_metrics/",
        "/home/ariel/Downloads/exp8_metrics/",
    ]

    save_dir = "./charts/"

    pearsons, i = [], 1

    for dir in dirs:
        print(f"\nUsing directory: {dir}")

        selection, case_numbers, DL, DS, pearson_coef = main(dir, i, save_dir=save_dir)
        pearsons.append(pearson_coef)
        best_cases = case_numbers[selection["best"]]
        worst_cases = case_numbers[selection["worst"]]
        ambiguous_cases = case_numbers[selection["ambiguous"]]
        conflict1_cases = case_numbers[selection["conflict1"]]
        conflict2_cases = case_numbers[selection["conflict2"]]
        all_cases = case_numbers[selection["all"]]

        print(f"best_cases = {best_cases}")
        print(f"worst_cases = {worst_cases}")
        print(f"ambiguous_cases = {ambiguous_cases}")
        print(f"conflict1_cases = {conflict1_cases}")
        print(f"conflict2_cases = {conflict2_cases}")
        print(f"all_cases = {all_cases}")

        i += 1

    pearsons = np.array(pearsons)
    pearsons.sort()
    print(f"\npearsons = {pearsons}")
