"""Coklu kuyu dataset hazirlama ve model egitimi kodu."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from feature_engineering import (
    add_geometry_features,
    add_gr_missing_features,
    add_gr_rolling_features,
)
from training_data import FeatureFunction, TrainingData
from utils import get_item_by_id, train_folder

BASELINE_FEATURE_COLUMNS = [
    # Ham ölçümler
    "MD",
    "X",
    "Y",
    "Z",
    "GR",

    # Yerel kuyu yonu ve egimi
    "dMD",
    "dX",
    "dY",
    "dZ",
    "dx_per_md",
    "dy_per_md",
    "dz_per_md",
    "horizontal_step",
    "inclination_deg",
    "azimuth_sin",
    "azimuth_cos",
    "dogleg_angle_deg",

    # GR ve eksiklik bilgisi
    "GR_isna",
    "GR_missing_run_length",
    "steps_since_valid_GR",
    "steps_until_valid_GR",

    # Yerel GR pattern'i
    "GR_roll_mean_10",
    "GR_roll_std_10",
    "GR_roll_range_10",
    "GR_missing_fraction_10",
    "GR_roll_mean_50",
    "GR_roll_std_50",
    "GR_roll_range_50",
    "GR_missing_fraction_50",
]

RNN_FEATURE_COLUMNS = [
    # Ham ölçümler
    "MD",
    "X",
    "Y",
    "Z",
    "GR",

    # Ardışık geometrik değişimler
    "dMD",
    "dX",
    "dY",
    "dZ",
    "dx_per_md",
    "dy_per_md",
    "dz_per_md",
    "horizontal_step",
    "inclination_deg",
    "azimuth_sin",
    "azimuth_cos",
    "dogleg_angle_deg",

    # GR güvenilirlik bilgisi
    "GR_isna",
    "GR_typewell_imputed_mask",
    "GR_neighbor_imputed_mask",
]

RNN_DROP_COLUMNS = [
    "ANCC",
    "ASTNU",
    "ASTNL",
    "EGFDU",
    "EGFDL",
    "BUDA",
    "TVT_input",
]


def make_baseline_feature_functions(
    _well: TrainingData,
) -> list[FeatureFunction]:
    """Global TVT tree modeli icin trajectory ve GR feature'larini kurar.

    Ne isimize yarar:
    Model PS'ye gore delta degil, her satirdaki mutlak TVT'yi ogrenir. Bu
    nedenle feature'lar yalnizca her satirda bulunan trajectory ve GR
    verisinden turetilir; PS veya TVT_input kullanilmaz.
    """
    return [
        add_geometry_features,
        add_gr_missing_features,
        lambda df: add_gr_rolling_features(
            df,
            window_sizes=(10, 50),
            center=True,
        ),
    ]

def make_rnn_feature_functions(
    _well: TrainingData,
) -> list[FeatureFunction]:
    return [
        add_geometry_features,
    ]


def prepare_rnn_features(
    training_ids: list[str],
    feature_columns: list[str],
    drop_columns: list[str],
    imputed: bool = True,
    data_dir: Path = train_folder,
    target_data_dir: Path | None = None,
) -> dict[str, TrainingData]:
    """Her train kuyusunu RNN feature'lariyla ayri ayri hazirlar."""
    prepared_wells: dict[str, TrainingData] = {}
    required_columns = [*feature_columns, "TVT"]

    for well_id in training_ids:
        well_data = TrainingData.from_well_id(
            well_id=well_id,
            data_dir=data_dir,
            feature_columns=feature_columns,
            imputed=imputed,
        ).prepare_base()

        if "TVT" not in well_data.data.columns:
            if target_data_dir is None:
                raise ValueError(
                    f"{well_id} verisinde TVT yok; offline skor icin "
                    "target_data_dir gerekli."
                )
            target_horizontal = get_item_by_id(
                well_id,
                target_data_dir,
            ).horizontal_well
            if target_horizontal["MD"].duplicated().any():
                raise ValueError(f"{well_id} target verisinde tekrarlanan MD var.")
            tvt_by_md = target_horizontal.set_index("MD")["TVT"]
            well_data.data["TVT"] = well_data.data["MD"].map(tvt_by_md)

        well_data.add_features(make_rnn_feature_functions(well_data))
        well_data.data["observed_TVT"] = well_data.data["TVT_input"]
        well_data.data["is_prediction_row"] = (
            well_data.data["TVT_input"].isna()
        )
        well_data.data = well_data.data.drop(
            columns=drop_columns,
            errors="ignore",
        )

        missing_columns = set(required_columns) - set(well_data.data.columns)
        if missing_columns:
            raise ValueError(
                f"{well_id} icin eksik RNN kolonlari: {sorted(missing_columns)}"
            )

        model_values = well_data.data[required_columns].to_numpy(dtype=float)
        valid_rows = np.isfinite(model_values).all(axis=1)
        well_data.data = (
            well_data.data.loc[valid_rows]
            .reset_index(drop=True)
            .copy()
        )
        prepared_wells[well_id] = well_data

    return prepared_wells


class RNNTrainingDataset(Dataset):
    """Tek bir kuyudan sabit uzunluklu RNN window'lari uretir."""

    def __init__(
        self,
        data: pd.DataFrame,
        window_size: int,
        feature_columns: list[str],
        target_column: str = "TVT",
        score_mask_column: str | None = None,
    ):
        if window_size <= 0:
            raise ValueError("window_size pozitif olmali.")

        required_columns = [*feature_columns, target_column]
        if score_mask_column is not None:
            required_columns.append(score_mask_column)
        missing_columns = set(required_columns) - set(data.columns)
        if missing_columns:
            raise ValueError(
                f"Dataset icin eksik kolonlar: {sorted(missing_columns)}"
            )

        self.data = data.sort_values("MD").reset_index(drop=True)
        self.window_size = window_size

        X = self.data[feature_columns].to_numpy(dtype=np.float32).copy()
        y = self.data[target_column].to_numpy(dtype=np.float32).copy()

        if not np.isfinite(X).all():
            raise ValueError("RNN feature'larinda NaN veya inf var.")
        if not np.isfinite(y).all():
            raise ValueError(f"{target_column} kolonunda NaN veya inf var.")

        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
        self.score_mask = (
            torch.from_numpy(
                self.data[score_mask_column]
                .to_numpy(dtype=bool)
                .copy()
            )
            if score_mask_column is not None
            else None
        )

    def __len__(self) -> int:
        return max(0, len(self.X) - self.window_size + 1)

    def __getitem__(self, idx: int):
        end_idx = idx + self.window_size
        X = self.X[idx:end_idx]
        y = self.y[end_idx - 1]
        if self.score_mask is not None:
            score_mask = self.score_mask[end_idx - 1]
            return X, y, score_mask
        return X, y


class MaskedTVTSequenceDataset(Dataset):
    """Her uygun satiri anchor yapan, maskeli TVT sequence dataset'i.

    `synthetic` modunda her index farkli bir anchor'a karsilik gelir. Anchor'a
    kadar gercek TVT context olarak verilir, sonraki sabit horizon maskelenir
    ve loss bolgesi olur. `real` modunda tek ornek uretilir; yalnizca gercek
    TVT_input prefix'i context, yarismadaki PS sonrasi bolge loss alanidir.

    Mevcut `RNNTrainingDataset`ten bagimsizdir ve onun davranisini degistirmez.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        feature_columns: list[str],
        target_column: str = "TVT_scaled",
        context_length: int = 64,
        forecast_length: int = 64,
        anchor_stride: int = 1,
        mode: str = "synthetic",
        prediction_mask_column: str = "is_prediction_row",
        observed_context_column: str = "observed_TVT_scaled",
    ):
        if context_length <= 0 or forecast_length <= 0:
            raise ValueError("context_length ve forecast_length pozitif olmali.")
        if anchor_stride <= 0:
            raise ValueError("anchor_stride pozitif olmali.")
        if mode not in {"synthetic", "real"}:
            raise ValueError("mode 'synthetic' veya 'real' olmali.")

        required_columns = [*feature_columns, target_column]
        if mode == "real":
            required_columns.extend(
                [prediction_mask_column, observed_context_column]
            )
        missing_columns = set(required_columns) - set(data.columns)
        if missing_columns:
            raise ValueError(
                f"Masked dataset icin eksik kolonlar: {sorted(missing_columns)}"
            )

        self.data = data.sort_values("MD").reset_index(drop=True)
        self.context_length = context_length
        self.forecast_length = forecast_length
        self.anchor_stride = anchor_stride
        self.mode = mode

        features = self.data[feature_columns].to_numpy(
            dtype=np.float32
        ).copy()
        target = self.data[target_column].to_numpy(dtype=np.float32).copy()
        if not np.isfinite(features).all():
            raise ValueError("Masked RNN feature'larinda NaN veya inf var.")
        if not np.isfinite(target).all():
            raise ValueError(f"{target_column} kolonunda NaN veya inf var.")

        self.features = torch.from_numpy(features)
        self.target = torch.from_numpy(target)

        if mode == "synthetic":
            self.first_anchor = context_length - 1
            self.last_anchor = len(self.data) - forecast_length - 1
            self.prediction_mask = None
            self.observed_context = None
        else:
            prediction_mask = self.data[prediction_mask_column].to_numpy(
                dtype=bool
            ).copy()
            observed_context = self.data[observed_context_column].to_numpy(
                dtype=np.float32
            ).copy()
            if not np.isfinite(observed_context).all():
                raise ValueError(
                    f"{observed_context_column} kolonunda NaN veya inf var."
                )
            self.prediction_mask = torch.from_numpy(prediction_mask)
            self.observed_context = torch.from_numpy(observed_context)
            self.first_anchor = -1
            self.last_anchor = -1

    def __len__(self) -> int:
        if self.mode == "real":
            return 1
        if self.last_anchor < self.first_anchor:
            return 0
        return (
            (self.last_anchor - self.first_anchor) // self.anchor_stride
        ) + 1

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | int]:
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)

        if self.mode == "synthetic":
            anchor = self.first_anchor + idx * self.anchor_stride
            start = anchor - self.context_length + 1
            end = anchor + self.forecast_length + 1

            base_features = self.features[start:end]
            target = self.target[start:end]
            known_mask = torch.zeros(len(target), dtype=torch.bool)
            known_mask[: self.context_length] = True
            loss_mask = ~known_mask

            tvt_context = torch.zeros_like(target)
            tvt_context[known_mask] = target[known_mask]
        else:
            base_features = self.features
            target = self.target
            loss_mask = self.prediction_mask.clone()
            known_mask = ~loss_mask
            tvt_context = self.observed_context.clone()
            tvt_context[loss_mask] = 0.0
            known_indices = torch.where(known_mask)[0]
            anchor = int(known_indices[-1]) if len(known_indices) else -1

        model_input = torch.cat(
            [
                base_features,
                tvt_context.unsqueeze(-1),
                known_mask.to(torch.float32).unsqueeze(-1),
            ],
            dim=-1,
        )

        return {
            "features": model_input,
            "target": target,
            "known_mask": known_mask,
            "loss_mask": loss_mask,
            "anchor": anchor,
        }


class StatefulTVTWindowDataset(Dataset):
    """Bir kuyuyu sirali, hidden-state tasiyan RNN parcalarina ayirir.

    `synthetic` modunda `prediction_start` sentetik PS'tir. Bu satirdan onceki
    TVT degerleri context olarak verilir, sonraki degerler sifirlanir ve loss
    bolgesi olur. `real` modunda gercek `TVT_input` maskesi kullanilir.

    Ilk loss parcasi `context_length + window_size` uzunlugundadir. Boylece
    model ilk tahminlere geri yayilim yaparken yakin TVT prefix'ini de gorur.
    Daha eski prefix, `window_size` uzunluklu isinma parcalariyla hidden state'e
    aktarilir. Sonraki tahmin parcalari sabit `window_size` ile devam eder.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        feature_columns: list[str],
        target_column: str = "TVT_scaled",
        window_size: int = 256,
        context_length: int = 256,
        mode: str = "synthetic",
        prediction_start: int | None = None,
        prediction_mask_column: str = "is_prediction_row",
        observed_context_column: str = "observed_TVT_scaled",
    ):
        if window_size <= 0 or context_length <= 0:
            raise ValueError("window_size ve context_length pozitif olmali.")
        if mode not in {"synthetic", "real"}:
            raise ValueError("mode 'synthetic' veya 'real' olmali.")

        required_columns = [*feature_columns, target_column]
        if mode == "real":
            required_columns.extend(
                [prediction_mask_column, observed_context_column]
            )
        missing_columns = set(required_columns) - set(data.columns)
        if missing_columns:
            raise ValueError(
                "Stateful dataset icin eksik kolonlar: "
                f"{sorted(missing_columns)}"
            )

        self.data = data.sort_values("MD").reset_index(drop=True)
        self.window_size = window_size
        self.context_length = context_length
        self.mode = mode

        features = self.data[feature_columns].to_numpy(
            dtype=np.float32
        ).copy()
        target = self.data[target_column].to_numpy(dtype=np.float32).copy()
        if not np.isfinite(features).all():
            raise ValueError("Stateful RNN feature'larinda NaN veya inf var.")
        if not np.isfinite(target).all():
            raise ValueError(f"{target_column} kolonunda NaN veya inf var.")

        num_rows = len(self.data)
        if mode == "synthetic":
            if prediction_start is None:
                raise ValueError(
                    "Synthetic modda prediction_start verilmelidir."
                )
            if prediction_start < context_length:
                raise ValueError(
                    "prediction_start en az context_length kadar olmali."
                )
            if prediction_start >= num_rows:
                raise ValueError("prediction_start kuyu uzunlugundan kucuk olmali.")

            known_mask = np.arange(num_rows) < prediction_start
            loss_mask = ~known_mask
            tvt_context = np.zeros(num_rows, dtype=np.float32)
            tvt_context[known_mask] = target[known_mask]
            self.prediction_start = prediction_start
        else:
            loss_mask = self.data[prediction_mask_column].to_numpy(
                dtype=bool
            ).copy()
            prediction_indices = np.flatnonzero(loss_mask)
            if len(prediction_indices) == 0:
                raise ValueError("Gercek PS sonrasinda tahmin satiri yok.")

            self.prediction_start = int(prediction_indices[0])
            expected_suffix = np.arange(self.prediction_start, num_rows)
            if not np.array_equal(prediction_indices, expected_suffix):
                raise ValueError(
                    "Gercek prediction mask kesintisiz bir suffix olmali."
                )

            known_mask = ~loss_mask
            tvt_context = self.data[observed_context_column].to_numpy(
                dtype=np.float32
            ).copy()
            tvt_context[loss_mask] = 0.0
            if not np.isfinite(tvt_context).all():
                raise ValueError(
                    f"{observed_context_column} kolonunda NaN veya inf var."
                )

        model_input = np.concatenate(
            [
                features,
                tvt_context[:, None],
                known_mask.astype(np.float32)[:, None],
            ],
            axis=1,
        )
        self.features = torch.from_numpy(model_input)
        self.target = torch.from_numpy(target)
        self.loss_mask = torch.from_numpy(loss_mask.copy())
        self.segments = self._make_segments(num_rows)

    def _make_segments(self, num_rows: int) -> list[tuple[int, int]]:
        context_start = max(0, self.prediction_start - self.context_length)
        segments: list[tuple[int, int]] = []

        for start in range(0, context_start, self.window_size):
            segments.append((start, min(start + self.window_size, context_start)))

        first_prediction_end = min(
            num_rows,
            self.prediction_start + self.window_size,
        )
        segments.append((context_start, first_prediction_end))

        for start in range(
            first_prediction_end,
            num_rows,
            self.window_size,
        ):
            segments.append((start, min(start + self.window_size, num_rows)))

        return segments

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | int]:
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        start, end = self.segments[idx]
        return {
            "features": self.features[start:end],
            "target": self.target[start:end],
            "loss_mask": self.loss_mask[start:end],
            "start": start,
            "end": end,
            "prediction_start": self.prediction_start,
        }


def build_training_dataset(
    well_ids: list[str],
    feature_columns: list[str],
    feature_function_factory: Callable[[TrainingData], list[FeatureFunction]],
    data_dir: Path = train_folder,
    post_ps_only: bool = False,
    imputed: bool = False,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Train veya validation kuyularini tek bir X, y ve group setine cevirir.

    Ne isimize yarar:
    Her kuyu ayri ayri feature engineering'den gecer; ardindan tree modelin
    egitimi icin secilen satirlar tek dataframe'de birlestirilir. Varsayilan
    tum kuyu boyunca mutlak TVT egitimidir. `post_ps_only=True` yalnizca
    Kaggle skor bolgesinde validation/test dataframe'i hazirlar. groups,
    satirlarin hangi kuyudan geldigini korur.
    """
    if not well_ids:
        raise ValueError("build_training_dataset() icin en az bir well ID gerekli.")

    X_parts = []
    y_parts = []
    group_parts = []

    for well_id in well_ids:
        well = TrainingData.from_well_id(
            well_id=well_id,
            data_dir=data_dir,
            feature_columns=feature_columns,
            imputed=imputed,
        ).prepare_base()

        well.add_features(feature_function_factory(well))
        if post_ps_only:
            well.set_observed_prefix_end()
        X_well = well.get_X(post_ps_only=post_ps_only)
        y_well = well.get_y(post_ps_only=post_ps_only)

        X_parts.append(X_well)
        y_parts.append(y_well)
        group_parts.append(
            pd.Series(well_id, index=X_well.index, name="well_id")
        )

    X = pd.concat(X_parts, ignore_index=True)
    y = pd.concat(y_parts, ignore_index=True).rename("TVT")
    groups = pd.concat(group_parts, ignore_index=True)
    return X, y, groups
