from dataclasses import dataclass
from pathlib import Path
import random

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

data_path = Path.cwd() / "data"
train_folder = data_path / "train"
test_folder = data_path / "test"
imputed_training_folder = data_path / "imputed_training_data"
imputed_test_folder = data_path / "imputed_test_data"

SEQUENCE_LENGTH = 64
WINDOW_STRIDE = 16
MAX_MASK_LENGTH = 19

GR_IMPUTATION_FEATURE_COLUMNS = [
    "MD",
    "X",
    "Y",
    "Z",
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
]

GEOMETRY_DERIVED_COLUMNS = [
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
]


class GRImputationDataset(Dataset):
    """Kuyu GR serilerinden yapay maskeli sabit uzunluklu windows uretir."""

    def __init__(
        self,
        well_data: dict[str, pd.DataFrame],
        sequence_length: int = SEQUENCE_LENGTH,
        window_stride: int = WINDOW_STRIDE,
        max_mask_length: int = MAX_MASK_LENGTH,
        random_state: int | None = None,
    ):
        self.sequence_length = sequence_length
        self.window_stride = window_stride
        self.max_mask_length = max_mask_length
        self.random_state = random_state
        self.well_data: dict[str, pd.DataFrame] = {}
        self.window_index: list[tuple[str, int]] = []

        for well_id, df in well_data.items():
            required_columns = {"MD", "GR"}
            missing_columns = required_columns - set(df.columns)
            if missing_columns:
                raise ValueError(
                    f"{well_id} icin eksik kolonlar: {sorted(missing_columns)}"
                )

            sequence = (
                df[["MD", "GR"]]
                .sort_values("MD")
                .reset_index(drop=True)
                .copy()
            )
            sequence["dMD"] = sequence["MD"].diff().fillna(0.0)

            if len(sequence) < self.sequence_length:
                continue

            self.well_data[well_id] = sequence
            starts = list(
                range(
                    0,
                    len(sequence) - self.sequence_length + 1,
                    self.window_stride,
                )
            )

            last_start = len(sequence) - self.sequence_length
            if starts[-1] != last_start:
                starts.append(last_start)

            for start in starts:
                window = sequence.iloc[start : start + self.sequence_length]
                if window["GR"].notna().any():
                    self.window_index.append((well_id, start))

    def __len__(self) -> int:
        return len(self.window_index)

    def _sample_synthetic_mask(
        self,
        observed_mask: np.ndarray,
        index: int,
    ) -> np.ndarray:
        seed = None if self.random_state is None else self.random_state + index
        rng = np.random.default_rng(seed)
        observed_indices = np.flatnonzero(observed_mask)
        split_points = np.where(np.diff(observed_indices) != 1)[0] + 1
        observed_runs = np.split(observed_indices, split_points)

        selected_run = observed_runs[rng.integers(len(observed_runs))]
        mask_length = rng.integers(
            1,
            min(self.max_mask_length, len(selected_run)) + 1,
        )
        start_offset = rng.integers(len(selected_run) - mask_length + 1)
        selected_indices = selected_run[start_offset : start_offset + mask_length]

        synthetic_mask = np.zeros(len(observed_mask), dtype=bool)
        synthetic_mask[selected_indices] = True
        return synthetic_mask

    def __getitem__(self, index: int) -> dict[str, np.ndarray | str | int]:
        well_id, start = self.window_index[index]
        window = self.well_data[well_id].iloc[
            start : start + self.sequence_length
        ].copy()

        gr_target = window["GR"].to_numpy(dtype=np.float32)
        observed_mask = ~np.isnan(gr_target)
        synthetic_mask = self._sample_synthetic_mask(observed_mask, index)

        loss_mask = synthetic_mask.astype(np.float32)
        input_observed_mask = observed_mask & ~synthetic_mask
        gr_input = np.where(input_observed_mask, gr_target, 0.0)

        # At t the model sees GR from t - 1, never GR[t].
        gr_history = np.roll(gr_input, 1)
        gr_history[0] = 0.0
        gr_history_mask = np.roll(input_observed_mask, 1).astype(np.float32)
        gr_history_mask[0] = 0.0

        features = np.column_stack(
            [
                window["dMD"].to_numpy(dtype=np.float32),
                gr_history,
                gr_history_mask,
            ]
        ).astype(np.float32)

        return {
            "features": features,
            "target": np.nan_to_num(gr_target, nan=0.0),
            "loss_mask": loss_mask,
            "well_id": well_id,
            "start": start,
        }


@dataclass(frozen=True)
class TrainFiles:
    id: str
    horizontal_well: Path
    typewell: Path
    png: Path


def get_all_ids(data_dir: Path) -> list[str]:
    """Verilen train veya test klasorundeki tum well ID'lerini dondurur."""
    return sorted(
        {
            file.name.split("__")[0]
            for file in data_dir.glob("*__horizontal_well.csv")
        }
    )


def get_all_training_ids() -> list[str]:
    """Train klasorundeki tum well ID'lerini dondurur."""
    return get_all_ids(train_folder)


def get_all_test_ids() -> list[str]:
    """Test klasorundeki tum well ID'lerini dondurur."""
    return get_all_ids(test_folder)

def get_item_by_id(well_id: str, train_dir: Path = train_folder) -> TrainFiles:
    return TrainFiles(
        id=well_id,
        png=train_dir / f"{well_id}.png",
        horizontal_well=pd.read_csv(train_dir / f"{well_id}__horizontal_well.csv"),
        typewell=pd.read_csv(train_dir / f"{well_id}__typewell.csv"),
        )

def get_random_training_file() -> TrainFiles:
    return get_item_by_id(random.choice(get_all_training_ids()))


def make_gr_ridge_pipeline(alpha: float = 10.0) -> Pipeline:
    """Her kuyu icin yeni bir GR Ridge regresyon pipeline'i olusturur."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=alpha)),
        ]
    )


def build_gr_imputation_X(df: pd.DataFrame) -> pd.DataFrame:
    """GR tahmininde kullanilacak, NaN icermeyen feature tablosunu kurar."""
    X = df[GR_IMPUTATION_FEATURE_COLUMNS].copy()
    X[GEOMETRY_DERIVED_COLUMNS] = (
        X[GEOMETRY_DERIVED_COLUMNS]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    if X.isna().any().any():
        missing_columns = X.columns[X.isna().any()].tolist()
        raise ValueError(f"GR imputation feature'larinda NaN var: {missing_columns}")

    return X


def impute_gr_with_ridge(
    df: pd.DataFrame,
    alpha: float = 10.0,
) -> tuple[pd.DataFrame, Pipeline]:
    """Bir kuyunun GR eksiklerini kendi dolu GR satirlariyla tahmin eder.

    Ham `GR` kolonu korunur. Tahminler `GR_ridge_prediction`, sadece eksik
    satirlara uygulanmis degerler ise `GR_ridge_imputed` kolonunda dondurulur.
    """
    result = df.copy()
    X_all = build_gr_imputation_X(result)
    known_gr_mask = result["GR"].notna()
    missing_gr_mask = result["GR"].isna()

    if known_gr_mask.sum() < 10:
        raise ValueError("Ridge fit etmek icin yeterli dolu GR satiri yok.")

    model = make_gr_ridge_pipeline(alpha=alpha)
    model.fit(
        X_all.loc[known_gr_mask],
        result.loc[known_gr_mask, "GR"],
    )

    result["GR_ridge_prediction"] = model.predict(X_all)
    result["GR_ridge_imputed"] = result["GR"]
    result.loc[missing_gr_mask, "GR_ridge_imputed"] = result.loc[
        missing_gr_mask, "GR_ridge_prediction"
    ]
    result["GR_isna"] = missing_gr_mask.astype("int8")
    return result, model


def evaluate_gr_ridge_imputation(
    df: pd.DataFrame,
    block_size: int = 100,
    alpha: float = 10.0,
    random_state: int = 42,
) -> dict[str, float | int]:
    """Dolu bir ardısik GR blogunu gizleyerek Ridge imputasyonunu olcer.

    Gercek GR NaN satirlarinin hedefi bilinmedigi icin kalite dogrudan
    olculemez. Bu fonksiyon, gozlemlenmis bir blogu gecici olarak maskeler;
    modeli kalan GR olcumleriyle fit eder ve gizlenen bloktaki RMSE/MAE'yi
    hesaplar.
    """
    known_gr_mask = df["GR"].notna()
    run_ids = known_gr_mask.ne(known_gr_mask.shift()).cumsum()
    known_runs = [
        run.index
        for _, run in df.loc[known_gr_mask].groupby(run_ids.loc[known_gr_mask])
    ]

    if not known_runs:
        raise ValueError(
            "Degerlendirme icin hic dolu GR satiri yok."
        )

    effective_block_size = min(block_size, max(map(len, known_runs)))
    if effective_block_size < 2:
        raise ValueError("Degerlendirme icin en az iki ardısik dolu GR satiri gerekli.")

    valid_runs = [
        run for run in known_runs if len(run) >= effective_block_size
    ]
    rng = np.random.default_rng(random_state)
    selected_run = valid_runs[rng.integers(len(valid_runs))]
    start_offset = rng.integers(len(selected_run) - effective_block_size + 1)
    validation_index = selected_run[
        start_offset : start_offset + effective_block_size
    ]

    validation_data = df.copy()
    actual_gr = validation_data.loc[validation_index, "GR"].copy()
    validation_data.loc[validation_index, "GR"] = np.nan

    imputed_data, _ = impute_gr_with_ridge(validation_data, alpha=alpha)
    predicted_gr = imputed_data.loc[validation_index, "GR_ridge_imputed"]

    return {
        "n_masked": len(validation_index),
        "rmse": float(root_mean_squared_error(actual_gr, predicted_gr)),
        "mae": float(mean_absolute_error(actual_gr, predicted_gr)),
    }


def save_imputed_training_data(
    output_dir: Path | None = None,
    alpha: float = 10.0,
    evaluation_block_size: int = 100,
    random_state: int = 42,
) -> Path:
    """Tum train kuyularini GR imputasyonu yaparak duz CSV yapisinda kaydeder.

    Her kuyu once sentetik olarak maskelenmis ardısik bir GR blogunda
    degerlendirilir. Sonra nihai Ridge modeli tum mevcut GR olcumleriyle fit
    edilir, imputasyon uygulanir ve dosyalar yazilir. Horizontal dosyalarda
    `GR`, Ridge ile doldurulmus degerdir; `GR_isna` orijinal GR olcumunun
    eksik olup olmadigini saklar. Typewell dosyalari degismeden yazilir.
    """
    # Local import avoids a circular import because training_data imports utils.
    from feature_engineering import add_geometry_features
    from training_data import TrainingData

    output_dir = output_dir or data_path / "imputed_training_data"
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluation_rows = []

    for well_number, well_id in enumerate(get_all_training_ids()):
        well_data = TrainingData.from_well_id(
            well_id=well_id,
            data_dir=train_folder,
        ).prepare_base()

        geometric_data = add_geometry_features(well_data.data.copy())

        metrics = evaluate_gr_ridge_imputation(
            geometric_data,
            block_size=evaluation_block_size,
            alpha=alpha,
            random_state=random_state + well_number,
        )
        evaluation_rows.append({"well_id": well_id, **metrics})

        # Evaluation sonrasinda nihai model, tum gercek GR olcumleriyle tekrar
        # fit edilir. Gizlenen validation blogu bu fit islemine dahil edilir.
        imputed_data, _ = impute_gr_with_ridge(geometric_data, alpha=alpha)

        imputed_data["GR"] = imputed_data["GR_ridge_imputed"]
        imputed_data = imputed_data.drop(
            columns=["GR_ridge_prediction", "GR_ridge_imputed"]
        )

        imputed_data.to_csv(
            output_dir / f"{well_id}__horizontal_well.csv",
            index=False,
        )
        well_data.typewell_df.to_csv(
            output_dir / f"{well_id}__typewell.csv",
            index=False,
        )

    evaluation = pd.DataFrame(evaluation_rows)
    evaluation.to_csv(output_dir / "gr_imputation_evaluation.csv", index=False)

    return output_dir
