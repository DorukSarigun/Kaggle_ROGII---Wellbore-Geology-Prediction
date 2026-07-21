"""ROGII wellbore geology prediction icin feature engineering fonksiyonlari."""

import numpy as np
import pandas as pd


def set_row_idx(df: pd.DataFrame) -> pd.DataFrame:
    """MD'ye gore siralar ve kuyu icinde stabil bir satir indeksi ekler.

    Ne isimize yarar:
    PS'nin satir konumunu, PS'den sonraki adim sayisini ve GR lag/lead
    feature'larini her kuyuda ayni referansa gore hesaplamamizi saglar.
    """
    df = df.sort_values("MD", ascending=True).reset_index(drop=True).copy()
    df["row_index"] = np.arange(len(df))
    return df


def get_ps_idx(df: pd.DataFrame) -> int:
    """TVT_input'un son dolu oldugu satiri, yani Prediction Start'i bulur.

    Ne isimize yarar:
    Bu satir son bilinen gercek TVT'dir. Bundan sonraki tum satirlar modelin
    tahmin edecegi bolgedir; PS yanlis bulunursa tum feature'lar kayar.
    """
    if not df["TVT_input"].notna().any():
        raise ValueError("PS bulunamadi: TVT_input tamamen bos.")
    return int(df["TVT_input"].last_valid_index())


def add_ps_position_features(df: pd.DataFrame, ps_idx: int) -> pd.DataFrame:
    """PS'ye ve kuyu sonuna gore satirin konumunu ekler.

    Ne isimize yarar:
    TVT davranisi PS'den ne kadar uzaklasildigina gore degisebilir. Model hem
    tahmin ufkunu hem de kuyunun sonuna ne kadar kaldigini gorur.
    """
    df["ps_idx"] = ps_idx
    df["steps_after_ps"] = df["row_index"] - ps_idx
    df["is_after_ps"] = (df["steps_after_ps"] > 0).astype("int8")
    df["total_steps_after_ps"] = len(df) - ps_idx - 1
    denominator = max(len(df) - ps_idx - 1, 1)
    df["progress_after_ps"] = df["steps_after_ps"] / denominator
    df["steps_to_end"] = len(df) - 1 - df["row_index"]
    return df


def add_md_features(df: pd.DataFrame, ps_idx: int) -> pd.DataFrame:
    """MD ekseninde PS'ye ve kuyu sonuna olan mesafeyi ekler.

    Ne isimize yarar:
    Satir sayisi sabit fiziksel mesafeyi temsil etmeyebilir. MD tabanli
    feature'lar, modelin kuyu boyunca gercek ilerlemeyi kullanmasini saglar.
    """
    md_at_ps = df.loc[ps_idx, "MD"]
    df["md_after_ps"] = df["MD"] - md_at_ps
    df["md_to_end"] = df["MD"].max() - df["MD"]
    total_md_after_ps = max(df["MD"].max() - md_at_ps, 1e-8)
    df["md_progress_after_ps"] = df["md_after_ps"] / total_md_after_ps
    return df


def add_spatial_ps_features(df: pd.DataFrame, ps_idx: int) -> pd.DataFrame:
    """XYZ koordinatlarinin PS noktasina gore kumeatif degisimini ekler.

    Ne isimize yarar:
    Kuyu PS'den sonra hangi yone ve ne kadar ilerledigini gosterir. Yerel
    jeolojik egim ve TVT degisimi bu hareketle iliskili olabilir.
    """
    ps_row = df.loc[ps_idx]
    df["x_after_ps"] = df["X"] - ps_row["X"]
    df["y_after_ps"] = df["Y"] - ps_row["Y"]
    df["z_after_ps"] = df["Z"] - ps_row["Z"]
    df["horizontal_distance_after_ps"] = np.hypot(
        df["x_after_ps"], df["y_after_ps"]
    )
    df["distance_3d_after_ps"] = np.sqrt(
        df["x_after_ps"] ** 2
        + df["y_after_ps"] ** 2
        + df["z_after_ps"] ** 2
    )
    return df


def add_ps_tvt_features(
    df: pd.DataFrame,
    ps_idx: int,
    slope_windows: tuple[int, ...] = (5, 20, 50, 100),
) -> pd.DataFrame:
    """PS oncesi bilinen TVT serisinden anchor ve trend feature'lari uretir.

    Ne isimize yarar:
    Son bilinen TVT tahmin icin en guclu baslangic referansidir. Farkli
    pencerelerdeki egim ve oynaklik, PS'ye girerkenki yerel jeolojik trendi
    modele verir; TVT'nin kendisini PS sonrasinda kullanmaz.
    """
    known = df.loc[:ps_idx, ["MD", "TVT_input"]].dropna()
    last_known_tvt = float(df.loc[ps_idx, "TVT_input"])
    df["last_known_TVT"] = last_known_tvt
    df["known_TVT_min"] = known["TVT_input"].min()
    df["known_TVT_max"] = known["TVT_input"].max()
    df["known_TVT_range"] = df["known_TVT_max"] - df["known_TVT_min"]
    df["last_TVT_change"] = known["TVT_input"].diff().iloc[-1]

    for window in slope_windows:
        tail = known.tail(window)
        if len(tail) >= 2 and tail["MD"].nunique() > 1:
            slope = np.polyfit(tail["MD"], tail["TVT_input"], 1)[0]
        else:
            slope = np.nan
        df[f"known_TVT_slope_{window}"] = slope
        df[f"known_TVT_std_{window}"] = tail["TVT_input"].std()
        df[f"known_TVT_range_{window}"] = (
            tail["TVT_input"].max() - tail["TVT_input"].min()
        )
    return df


def add_geometry_features(df: pd.DataFrame) -> pd.DataFrame:
    """Kuyunun yerel yonu, egimi ve yon degisimini hesaplar.

    Ne isimize yarar:
    dX/dY/dZ anlik hareketi, x/y/z_after_ps ise kumeatif hareketi anlatir.
    Inclination, azimuth ve dogleg ise kuyunun jeolojik katmanlari kesme
    bicimini daha kompakt olarak modele verir.
    """
    df["dMD"] = df["MD"].diff()
    df["dX"] = df["X"].diff()
    df["dY"] = df["Y"].diff()
    df["dZ"] = df["Z"].diff()

    valid_dmd = df["dMD"].replace(0, np.nan)
    df["dx_per_md"] = df["dX"] / valid_dmd
    df["dy_per_md"] = df["dY"] / valid_dmd
    df["dz_per_md"] = df["dZ"] / valid_dmd
    df["horizontal_step"] = np.hypot(df["dX"], df["dY"])

    vertical_ratio = np.clip(np.abs(df["dz_per_md"]), 0, 1)
    df["inclination_deg"] = np.degrees(np.arccos(vertical_ratio))
    azimuth = np.arctan2(df["dY"], df["dX"])
    df["azimuth_sin"] = np.sin(azimuth)
    df["azimuth_cos"] = np.cos(azimuth)

    direction = df[["dX", "dY", "dZ"]].to_numpy(dtype=float)
    norm = np.linalg.norm(direction, axis=1)
    unit = np.divide(
        direction,
        norm[:, None],
        out=np.full_like(direction, np.nan),
        where=norm[:, None] != 0,
    )
    dots = np.sum(unit[1:] * unit[:-1], axis=1)
    df["dogleg_angle_deg"] = np.r_[
        np.nan, np.degrees(np.arccos(np.clip(dots, -1, 1)))
    ]
    return df


def add_gr_missing_features(df: pd.DataFrame) -> pd.DataFrame:
    """GR eksikliklerini ve eksik bloklarin yapisini feature'a cevirir.

    Ne isimize yarar:
    GR'deki NaN degerleri rastgele olmayabilir. Missing flag ve eksik blok
    uzunlugu modelin GR sinyaline ne kadar guvenebilecegini anlamasina yardim
    eder. Interpolated GR ise lag/lead feature'lari icin kesintisiz seri verir.
    """
    missing = df["GR"].isna()
    df["GR_isna"] = missing.astype("int8")
    df["GR_interpolated"] = df["GR"].interpolate(
        method="linear", limit_direction="both"
    )

    groups = missing.ne(missing.shift()).cumsum()
    df["GR_missing_run_length"] = (
        missing.groupby(groups).transform("sum").where(missing, 0)
    )
    positions = pd.Series(np.arange(len(df)), index=df.index, dtype=float)
    previous_valid = positions.where(~missing).ffill()
    next_valid = positions.where(~missing).bfill()
    df["steps_since_valid_GR"] = (positions - previous_valid).where(missing, 0)
    df["steps_until_valid_GR"] = (next_valid - positions).where(missing, 0)
    return df


def add_gr_rolling_features(
    df: pd.DataFrame,
    window_sizes: tuple[int, ...] = (10, 50, 200),
    center: bool = True,
) -> pd.DataFrame:
    """GR'nin yerel ortalama, yayilim, aralik ve lokal z-score'unu ekler.

    Ne isimize yarar:
    Tekil GR degeri yerine kayacin yerel GR imzasini verir. Tum test GR
    dizisi dosyada mevcut oldugu icin centered pencere kullanmak uygundur;
    gelecekteki TVT kullanilmadigindan leakage olusturmaz.
    """
    gr = df["GR"]
    for window_size in window_sizes:
        rolling = gr.rolling(
            window=window_size, min_periods=1, center=center
        )
        mean = rolling.mean()
        std = rolling.std()
        df[f"GR_roll_mean_{window_size}"] = mean
        df[f"GR_roll_std_{window_size}"] = std
        df[f"GR_roll_max_{window_size}"] = rolling.max()
        df[f"GR_roll_min_{window_size}"] = rolling.min()
        df[f"GR_roll_range_{window_size}"] = (
            df[f"GR_roll_max_{window_size}"] - df[f"GR_roll_min_{window_size}"]
        )
        df[f"GR_roll_count_{window_size}"] = rolling.count()
        df[f"GR_missing_fraction_{window_size}"] = (
            df["GR_isna"]
            .rolling(window=window_size, min_periods=1, center=center)
            .mean()
        )
        df[f"GR_local_zscore_{window_size}"] = (
            (gr - mean) / std.replace(0, np.nan)
        )
    return df


def add_gr_sequence_features(
    df: pd.DataFrame,
    steps: tuple[int, ...] = (1, 5, 20),
) -> pd.DataFrame:
    """GR serisine lag, lead ve fark feature'lari ekler.

    Ne isimize yarar:
    GR'nin sadece seviyesi degil, artma/azalma bicimi de katman imzasidir.
    Lead feature'lari test dosyasindaki tum GR serisi basta mevcut oldugu icin
    kullanilabilir; TVT veya TVT_input'un geleceginden bilgi tasimaz.
    """
    gr = df["GR_interpolated"]
    df["GR_diff_1"] = gr.diff()
    for step in steps:
        df[f"GR_lag_{step}"] = gr.shift(step)
        df[f"GR_lead_{step}"] = gr.shift(-step)
        df[f"GR_diff_{step}"] = gr - gr.shift(step)
    return df


def add_typewell_features(
    df: pd.DataFrame,
    typewell_df: pd.DataFrame,
    ps_idx: int,
) -> pd.DataFrame:
    """Typewell GR-TVT referansindan kuyu seviyesinde feature'lar uretir.

    Ne isimize yarar:
    Yarismadaki temel sinyal, horizontal GR profilinin typewell GR profiline
    benzemesidir. PS oncesi bilinen TVT-GR ciftleriyle kuyuya ozel GR scale ve
    offset hesaplanir; model typewell ile ne kadar uyum oldugunu gorur.
    """
    typewell = typewell_df.sort_values("TVT").dropna(subset=["TVT", "GR"])
    typewell = typewell.drop_duplicates("TVT")
    tw_tvt = typewell["TVT"].to_numpy(dtype=float)
    tw_gr = typewell["GR"].to_numpy(dtype=float)
    if len(typewell) < 2:
        raise ValueError("Typewell icinde yeterli TVT-GR verisi yok.")

    anchor = float(df.loc[ps_idx, "TVT_input"])
    df["typewell_TVT_min"] = tw_tvt.min()
    df["typewell_TVT_max"] = tw_tvt.max()
    df["typewell_TVT_range"] = tw_tvt.max() - tw_tvt.min()
    df["distance_to_typewell_top"] = anchor - tw_tvt.min()
    df["distance_to_typewell_bottom"] = tw_tvt.max() - anchor
    df["typewell_GR_mean"] = np.mean(tw_gr)
    df["typewell_GR_std"] = np.std(tw_gr)
    df["typewell_GR_at_PS"] = np.interp(anchor, tw_tvt, tw_gr)

    tw_gradient = np.gradient(tw_gr, tw_tvt)
    df["typewell_GR_gradient_at_PS"] = np.interp(
        anchor, tw_tvt, tw_gradient
    )

    known = df.loc[:ps_idx, ["TVT_input", "GR"]].dropna()
    if len(known) >= 3:
        expected_gr = np.interp(known["TVT_input"], tw_tvt, tw_gr)
        observed_gr = known["GR"].to_numpy(dtype=float)
        scale, offset = np.polyfit(expected_gr, observed_gr, 1)
        fitted = scale * expected_gr + offset
        alignment_rmse = np.sqrt(np.mean((observed_gr - fitted) ** 2))
        correlation = np.corrcoef(expected_gr, observed_gr)[0, 1]
    else:
        scale, offset, alignment_rmse, correlation = np.nan, np.nan, np.nan, np.nan

    df["typewell_GR_scale"] = scale
    df["typewell_GR_offset"] = offset
    df["known_typewell_alignment_rmse"] = alignment_rmse
    df["known_typewell_alignment_corr"] = correlation
    if np.isfinite(scale) and abs(scale) > 1e-8:
        df["GR_calibrated_to_typewell"] = (df["GR"] - offset) / scale
    else:
        df["GR_calibrated_to_typewell"] = df["GR"]
    return df


TRAIN_ONLY_COLUMNS = {
    "TVT",
    "target_delta_tvt",
    "ANCC",
    "ASTNU",
    "ASTNL",
    "EGFDU",
    "EGFDL",
    "BUDA",
}
NON_FEATURE_COLUMNS = TRAIN_ONLY_COLUMNS | {"TVT_input", "is_after_ps"}


def get_model_feature_columns(feature_df: pd.DataFrame) -> list[str]:
    """Train ve testte ortak, sayisal ve leakage icermeyen kolonlari dondurur.

    Ne isimize yarar:
    TVT target'inin, TVT_input'un veya sadece train'de bulunan formasyon
    kolonlarinin LightGBM'e yanlislikla verilmesini engeller.
    """
    return [
        column
        for column in feature_df.select_dtypes(include=np.number).columns
        if column not in NON_FEATURE_COLUMNS
    ]


def build_features(
    horizontal_df: pd.DataFrame,
    typewell_df: pd.DataFrame,
    rolling_windows: tuple[int, ...] = (10, 50, 200),
    lag_steps: tuple[int, ...] = (1, 5, 20),
) -> pd.DataFrame:
    """Bir horizontal well ve typewell icin tum feature setini uretir.

    Ne isimize yarar:
    Train ve testte birebir ayni donusumleri tek noktadan calistirir. Boylece
    feature uyumsuzlugu azalir ve model pipeline'i tekrar uretilebilir kalir.
    """
    df = set_row_idx(horizontal_df)
    ps_idx = get_ps_idx(df)
    df = add_ps_position_features(df, ps_idx)
    df = add_md_features(df, ps_idx)
    df = add_spatial_ps_features(df, ps_idx)
    df = add_ps_tvt_features(df, ps_idx)
    df = add_geometry_features(df)
    df = add_gr_missing_features(df)
    df = add_gr_rolling_features(df, rolling_windows)
    df = add_gr_sequence_features(df, lag_steps)
    df = add_typewell_features(df, typewell_df, ps_idx)

    if "TVT" in df.columns:
        df["target_delta_tvt"] = df["TVT"] - df["last_known_TVT"]
    return df
