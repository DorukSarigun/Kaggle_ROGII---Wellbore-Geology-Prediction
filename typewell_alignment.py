"""Horizontal tahminleri typewell GR profiliyle row bazinda duzeltir."""

from pathlib import Path

import numpy as np
import pandas as pd

from feature_engineering import get_ps_idx, set_row_idx
from utils import get_item_by_id


def impute_gr_with_known_tvt_and_typewell(
    horizontal_df: pd.DataFrame,
    typewell_df: pd.DataFrame,
    min_calibration_points: int = 20,
    min_alignment_corr: float = 0.2,
) -> tuple[pd.DataFrame, dict[str, float | int | bool]]:
    """TVT_input bilinen satirlarda typewell GR ile horizontal GR doldurur.

    Typewell GR, horizontal `TVT_input` koordinatlarina interpolate edilir.
    Typewell ile horizontal GR olcek farki, iki GR de bilinen satirlardan
    `horizontal_GR = scale * typewell_GR + offset` olarak kalibre edilir.
    Gercek horizontal GR degerleri degistirilmez.
    """
    required_horizontal = {"GR", "TVT_input"}
    missing_horizontal = required_horizontal - set(horizontal_df.columns)
    if missing_horizontal:
        raise ValueError(
            f"Horizontal veride eksik kolonlar: {sorted(missing_horizontal)}"
        )

    required_typewell = {"TVT", "GR"}
    missing_typewell = required_typewell - set(typewell_df.columns)
    if missing_typewell:
        raise ValueError(
            f"Typewell veride eksik kolonlar: {sorted(missing_typewell)}"
        )

    result = horizontal_df.copy()
    original_missing_mask = result["GR"].isna()
    result["GR_isna"] = original_missing_mask.astype("int8")

    typewell = (
        typewell_df[["TVT", "GR"]]
        .dropna()
        .sort_values("TVT")
        .drop_duplicates("TVT")
    )
    if len(typewell) < 2:
        raise ValueError("Typewell GR-TVT kalibrasyonu icin yeterli veri yok.")

    typewell_tvt = typewell["TVT"].to_numpy(dtype=float)
    typewell_gr = typewell["GR"].to_numpy(dtype=float)
    horizontal_tvt = result["TVT_input"].to_numpy(dtype=float)
    typewell_gr_at_tvt = np.interp(
        horizontal_tvt,
        typewell_tvt,
        typewell_gr,
        left=np.nan,
        right=np.nan,
    )
    result["typewell_GR_at_TVT_input"] = typewell_gr_at_tvt

    calibration_mask = (
        result["GR"].notna()
        & result["TVT_input"].notna()
        & result["typewell_GR_at_TVT_input"].notna()
    )
    calibration_count = int(calibration_mask.sum())
    if calibration_count < min_calibration_points:
        raise ValueError(
            "Typewell GR kalibrasyonu icin yeterli horizontal GR-TVT_input "
            f"cifti yok: {calibration_count}"
        )

    expected_gr = result.loc[
        calibration_mask, "typewell_GR_at_TVT_input"
    ].to_numpy(dtype=float)
    observed_gr = result.loc[calibration_mask, "GR"].to_numpy(dtype=float)
    if np.std(expected_gr) < 1e-8 or np.std(observed_gr) < 1e-8:
        raise ValueError("GR kalibrasyonu icin GR varyasyonu yetersiz.")

    alignment_corr = float(np.corrcoef(expected_gr, observed_gr)[0, 1])
    scale, offset = np.polyfit(expected_gr, observed_gr, 1)
    scale = float(scale)
    offset = float(offset)
    calibrated_training_gr = scale * expected_gr + offset
    calibration_rmse = float(
        np.sqrt(np.mean((observed_gr - calibrated_training_gr) ** 2))
    )
    calibration_is_valid = bool(
        np.isfinite(alignment_corr)
        and np.isfinite(scale)
        and np.isfinite(offset)
        and alignment_corr >= min_alignment_corr
    )

    result["typewell_GR_prediction"] = (
        scale * result["typewell_GR_at_TVT_input"] + offset
    )
    fill_mask = (
        original_missing_mask
        & result["TVT_input"].notna()
        & result["typewell_GR_prediction"].notna()
        & calibration_is_valid
    )
    result["GR_typewell_imputed"] = result["GR"]
    result.loc[fill_mask, "GR_typewell_imputed"] = result.loc[
        fill_mask, "typewell_GR_prediction"
    ]
    result["GR_typewell_imputed_mask"] = fill_mask.astype("int8")

    diagnostics: dict[str, float | int | bool] = {
        "calibration_points": calibration_count,
        "alignment_corr": alignment_corr,
        "calibration_rmse": calibration_rmse,
        "scale": scale,
        "offset": offset,
        "calibration_is_valid": calibration_is_valid,
        "imputed_rows": int(fill_mask.sum()),
        "remaining_missing_rows": int(result["GR_typewell_imputed"].isna().sum()),
    }
    return result, diagnostics


def impute_remaining_gr_with_neighbor_mean(
    horizontal_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Her ic NaN GR blogunu ust ve alt komsu GR ortalamasiyla doldurur."""
    if "GR" not in horizontal_df.columns:
        raise ValueError("Horizontal veride GR kolonu yok.")

    result = horizontal_df.copy()
    gr = result["GR"].to_numpy(dtype=float, copy=True)
    missing_mask = np.isnan(gr)
    padded_mask = np.concatenate(([False], missing_mask, [False])).astype(int)
    transitions = np.diff(padded_mask)
    run_starts = np.flatnonzero(transitions == 1)
    run_ends = np.flatnonzero(transitions == -1) - 1
    imputed_mask = np.zeros(len(result), dtype=bool)
    max_block_length = 0
    one_sided_imputed_rows = 0

    for start, end in zip(run_starts, run_ends, strict=True):
        block_length = int(end - start + 1)
        max_block_length = max(max_block_length, block_length)
        left_position = start - 1
        right_position = end + 1

        left_value = gr[left_position] if left_position >= 0 else np.nan
        right_value = gr[right_position] if right_position < len(gr) else np.nan
        left_is_valid = np.isfinite(left_value)
        right_is_valid = np.isfinite(right_value)

        if left_is_valid and right_is_valid:
            fill_value = (left_value + right_value) / 2.0
        elif left_is_valid:
            fill_value = left_value
            one_sided_imputed_rows += block_length
        elif right_is_valid:
            fill_value = right_value
            one_sided_imputed_rows += block_length
        else:
            continue

        gr[start : end + 1] = fill_value
        imputed_mask[start : end + 1] = True

    result["GR"] = gr
    result["GR_neighbor_imputed_mask"] = imputed_mask.astype("int8")
    diagnostics = {
        "neighbor_imputed_rows": int(imputed_mask.sum()),
        "one_sided_neighbor_imputed_rows": one_sided_imputed_rows,
        "max_neighbor_block_length": max_block_length,
        "final_missing_rows": int(np.isnan(gr).sum()),
    }
    return result, diagnostics


def save_known_tvt_typewell_gr_imputation(
    well_ids: list[str],
    source_dir: Path,
    output_dir: Path,
    min_calibration_points: int = 20,
    min_alignment_corr: float = 0.2,
) -> pd.DataFrame:
    """Typewell ile doldurulmus horizontal ve typewell CSV'lerini kaydeder."""
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_rows = []

    for well_id in well_ids:
        loaded_well = get_item_by_id(well_id, source_dir)
        try:
            imputed_data, diagnostics = impute_gr_with_known_tvt_and_typewell(
                horizontal_df=loaded_well.horizontal_well,
                typewell_df=loaded_well.typewell,
                min_calibration_points=min_calibration_points,
                min_alignment_corr=min_alignment_corr,
            )
            diagnostics["calibration_error"] = ""
        except ValueError as error:
            imputed_data = loaded_well.horizontal_well.copy()
            original_missing_mask = imputed_data["GR"].isna()
            imputed_data["GR_isna"] = original_missing_mask.astype("int8")
            imputed_data["GR_typewell_imputed"] = imputed_data["GR"]
            imputed_data["GR_typewell_imputed_mask"] = 0
            diagnostics = {
                "calibration_points": 0,
                "alignment_corr": np.nan,
                "calibration_rmse": np.nan,
                "scale": np.nan,
                "offset": np.nan,
                "calibration_is_valid": False,
                "imputed_rows": 0,
                "remaining_missing_rows": int(original_missing_mask.sum()),
                "calibration_error": str(error),
            }

        horizontal_output = loaded_well.horizontal_well.copy()
        horizontal_output["GR_isna"] = imputed_data["GR_isna"]
        horizontal_output["GR_typewell_imputed_mask"] = imputed_data[
            "GR_typewell_imputed_mask"
        ]
        horizontal_output["GR"] = imputed_data["GR_typewell_imputed"]
        horizontal_output, neighbor_diagnostics = (
            impute_remaining_gr_with_neighbor_mean(horizontal_output)
        )
        diagnostics.update(neighbor_diagnostics)

        horizontal_output.to_csv(
            output_dir / f"{well_id}__horizontal_well.csv",
            index=False,
        )
        loaded_well.typewell.to_csv(
            output_dir / f"{well_id}__typewell.csv",
            index=False,
        )
        diagnostics_rows.append({"well_id": well_id, **diagnostics})

    diagnostics_df = pd.DataFrame(diagnostics_rows)
    diagnostics_df.to_csv(
        output_dir / "typewell_gr_imputation_diagnostics.csv",
        index=False,
    )
    return diagnostics_df


def refine_predictions_with_typewell(
    X: pd.DataFrame,
    groups: pd.Series,
    pred_delta: np.ndarray,
    data_dir: Path,
    search_radius: float = 20.0,
    prior_penalty: float = 0.1,
    blend: float = 0.25,
    min_alignment_corr: float = 0.4,
) -> np.ndarray:
    """Baseline delta TVT tahminini typewell GR ile row bazinda duzeltir.

    Her horizontal row icin once baseline modelin mutlak TVT tahmini prior
    olarak alinir. Typewell'de bu prior'in yakinindaki TVT adaylari aranir ve
    horizontal GR'ye en uygun aday secilir. Son tahmin hard eslesme yerine
    prior ile aday arasinda kucuk bir blend yapilarak uretilir.

    PS sonrasi gercek TVT bu fonksiyonda kullanilmaz. GR kalibrasyonu sadece
    PS oncesindeki TVT_input ve GR ciftlerinden hesaplanir.
    """
    required_columns = {"GR", "last_known_TVT"}
    missing_columns = required_columns - set(X.columns)
    if missing_columns:
        raise ValueError(
            f"Typewell refinement icin eksik kolonlar: {sorted(missing_columns)}"
        )
    if not (len(X) == len(groups) == len(pred_delta)):
        raise ValueError("X, groups ve pred_delta ayni uzunlukta olmali.")
    if search_radius <= 0:
        raise ValueError("search_radius pozitif olmali.")
    if not 0 <= blend <= 1:
        raise ValueError("blend 0 ile 1 arasinda olmali.")

    group_values = groups.reset_index(drop=True).to_numpy()
    anchor_tvt = X["last_known_TVT"].to_numpy(dtype=float)
    prior_tvt = anchor_tvt + np.asarray(pred_delta, dtype=float)
    refined_tvt = prior_tvt.copy()

    for well_id in pd.unique(group_values):
        row_positions = np.flatnonzero(group_values == well_id)
        loaded_well = get_item_by_id(str(well_id), data_dir)

        horizontal = set_row_idx(loaded_well.horizontal_well)
        ps_idx = get_ps_idx(horizontal)
        typewell = (
            loaded_well.typewell.sort_values("TVT")
            .dropna(subset=["TVT", "GR"])
            .drop_duplicates("TVT")
        )
        if len(typewell) < 2:
            continue

        typewell_tvt = typewell["TVT"].to_numpy(dtype=float)
        typewell_gr = typewell["GR"].to_numpy(dtype=float)
        known = horizontal.loc[:ps_idx, ["TVT_input", "GR"]].dropna()
        if len(known) < 3:
            continue

        expected_gr = np.interp(
            known["TVT_input"].to_numpy(dtype=float),
            typewell_tvt,
            typewell_gr,
        )
        observed_gr = known["GR"].to_numpy(dtype=float)
        if np.std(expected_gr) == 0 or np.std(observed_gr) == 0:
            continue

        alignment_corr = np.corrcoef(expected_gr, observed_gr)[0, 1]
        if not np.isfinite(alignment_corr) or alignment_corr < min_alignment_corr:
            continue

        scale, offset = np.polyfit(expected_gr, observed_gr, 1)
        if not np.isfinite(scale) or not np.isfinite(offset) or abs(scale) < 1e-8:
            continue

        horizontal_gr = X.iloc[row_positions]["GR"].to_numpy(dtype=float)
        calibrated_gr = (horizontal_gr - offset) / scale

        for local_position, row_position in enumerate(row_positions):
            row_gr = calibrated_gr[local_position]
            if not np.isfinite(row_gr):
                continue

            row_prior_tvt = prior_tvt[row_position]
            candidate_mask = (
                np.abs(typewell_tvt - row_prior_tvt) <= search_radius
            )
            if not candidate_mask.any():
                continue

            candidate_tvt = typewell_tvt[candidate_mask]
            candidate_gr = typewell_gr[candidate_mask]
            candidate_cost = (
                np.abs(candidate_gr - row_gr)
                + prior_penalty * np.abs(candidate_tvt - row_prior_tvt)
            )
            matched_tvt = candidate_tvt[np.argmin(candidate_cost)]
            refined_tvt[row_position] = (
                row_prior_tvt + blend * (matched_tvt - row_prior_tvt)
            )

    return refined_tvt - anchor_tvt
