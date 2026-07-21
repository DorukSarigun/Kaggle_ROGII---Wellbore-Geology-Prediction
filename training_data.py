"""Model egitimi oncesi well verisini yoneten sinif."""

from collections.abc import Callable
from pathlib import Path

import pandas as pd

# from training_data import FeatureFunction, TrainingData
from utils import (
    get_item_by_id,
    imputed_test_folder,
    imputed_training_folder,
    test_folder,
    train_folder,
)
from feature_engineering import get_ps_idx, set_row_idx

FeatureFunction = Callable[[pd.DataFrame], pd.DataFrame]


class TrainingData:
    """Bir horizontal well ve ona ait typewell verisini hazirlar."""

    def __init__(
        self,
        horizontal_df: pd.DataFrame,
        typewell_df: pd.DataFrame,
        feature_columns: list[str] | None = None,
    ):
        """Ham veriyi kopyalayarak saklar.

        Ne isimize yarar:
        Feature engineering sirasinda orijinal dataframe'lerin degismemesini
        saglar. Ayni ham kuyu verisinden farkli feature denemeleri yapabiliriz.
        """
        self.horizontal_df = horizontal_df.copy()
        self.typewell_df = typewell_df.copy()
        self.data: pd.DataFrame | None = None
        self.ps_idx: int | None = None
        self.feature_columns = feature_columns.copy() if feature_columns else None

    def prepare_base(self) -> "TrainingData":
        """Horizontal well'i siralar ve feature calisma tablosunu kurar.

        Ne isimize yarar:
        `horizontal_df` ham veri olarak degismeden kalir. `data`, MD'ye gore
        siralanmis ve `row_index` eklenmis calisma kopyasidir; feature
        engineering bu tablo uzerinde yapilir. PS bilgisi burada hesaplanmaz,
        cunku sequence egitiminde her satir sentetik anchor olabilir.
        """
        self.data = set_row_idx(self.horizontal_df)
        return self

    def set_observed_prefix_end(self) -> "TrainingData":
        """Gercek `TVT_input` prefix'inin son satirini belirler.

        Ne isimize yarar:
        Bu sinir yalnizca Kaggle test/validation bolgesini secmek icindir.
        Model feature'i veya sequence egitimindeki sentetik anchor tanimi
        degildir.
        """
        if self.data is None:
            raise RuntimeError(
                "set_observed_prefix_end() oncesinde prepare_base() cagirilmali."
            )

        self.ps_idx = get_ps_idx(self.data)
        return self

    def add_features(
        self,
        feature_functions: list[FeatureFunction],
    ) -> "TrainingData":
        """Verilen feature fonksiyonlarini sirayla self.data uzerine uygular.

        Ne isimize yarar:
        Her deneyde kullanmak istedigimiz feature gruplarini liste olarak
        secebiliriz. Fonksiyonlarin `df -> df` formatinda dataframe dondurmesi
        gerekir; boylece sinif feature fonksiyonlarina bagimli kalmaz.
        """
        if self.data is None:
            raise RuntimeError("add_features() oncesinde prepare_base() cagirilmali.")

        for feature_function in feature_functions:
            self.data = feature_function(self.data)
        return self

    def set_feature_columns(self, feature_columns: list[str]) -> "TrainingData":
        """Tum kuyularda kullanilacak model feature kolonlarini kaydeder.

        Ne isimize yarar:
        Train ve testte ayni kolonlarin, ayni sirayla modele verilmesini
        saglar. Feature listesi bir kez train setup'ta belirlenmelidir.
        """
        self.feature_columns = feature_columns.copy()
        return self

    def get_post_ps_data(self) -> pd.DataFrame:
        """PS sonrasi, modelin tahmin yapacagi satirlari dondurur.

        Ne isimize yarar:
        Train ve testte ayni satir filtresini kullaniriz. PS satiri bilinen
        anchor oldugu icin target veya tahmin satirlarina dahil edilmez.
        """
        if self.data is None or self.ps_idx is None:
            raise RuntimeError(
                "get_post_ps_data() oncesinde set_observed_prefix_end() cagirilmali."
            )
        return self.data.loc[self.data["row_index"] > self.ps_idx].copy()

    def get_X(self, post_ps_only: bool = False) -> pd.DataFrame:
        """Secilen satirlar icin model feature dataframe'ini dondurur.

        Global TVT modeli tum kuyu boyunca egitilir; bu nedenle varsayilan
        tum satirlari dondurmektir. `post_ps_only=True` yalnizca Kaggle
        submission/validation bolgesini secmek icin kullanilir ve model
        feature'larina PS bilgisi eklemez.
        """
        if self.feature_columns is None:
            raise RuntimeError(
                "get_X() oncesinde set_feature_columns() cagirilmali."
            )

        if self.data is None:
            raise RuntimeError("get_X() oncesinde prepare_base() cagirilmali.")

        selected_data = (
            self.get_post_ps_data() if post_ps_only else self.data.copy()
        )
        missing_columns = set(self.feature_columns) - set(selected_data.columns)
        if missing_columns:
            raise ValueError(
                f"Feature kolonlari veride yok: {sorted(missing_columns)}"
            )
        return selected_data[self.feature_columns].copy()

    def get_y(self, post_ps_only: bool = False) -> pd.Series:
        """Train kuyusunun mutlak TVT hedefini dondurur.

        Model TVT'yi dogrudan ogrenir. `post_ps_only=True` yalnizca
        yarismadaki skor bolgesinde validation hesaplamak icindir.
        """
        if self.data is None:
            raise RuntimeError("get_y() oncesinde prepare_base() cagirilmali.")
        if "TVT" not in self.data.columns:
            raise ValueError("get_y() sadece TVT iceren train verisinde kullanilir.")

        selected_data = (
            self.get_post_ps_data() if post_ps_only else self.data
        )
        return selected_data["TVT"].copy()

    @classmethod
    def from_well_id(
        cls,
        well_id: str,
        data_dir: Path = train_folder,
        feature_columns: list[str] | None = None,
        imputed: bool = False,
    ) -> "TrainingData":
        """Well ID'den veri yukleyerek TrainingData nesnesi olusturur.

        Ne isimize yarar:
        `imputed=False` varsayilan ham veriyi okur. `imputed=True` ise train
        klasorunu `imputed_training_data`, test klasorunu `imputed_test_data`
        ile esler. Boylece cagiran kodun geri kalani degismeden iki veri
        kaynagi arasinda secim yapabilir.
        """
        selected_data_dir = data_dir
        if imputed:
            resolved_data_dir = data_dir.resolve()
            if resolved_data_dir == train_folder.resolve():
                selected_data_dir = imputed_training_folder
            elif resolved_data_dir == test_folder.resolve():
                selected_data_dir = imputed_test_folder
            elif resolved_data_dir not in {
                imputed_training_folder.resolve(),
                imputed_test_folder.resolve(),
            }:
                raise ValueError(
                    "imputed=True yalnizca standart train/test veya imputed "
                    "klasorleriyle kullanilabilir."
                )

        loaded_well = get_item_by_id(well_id, selected_data_dir)

        return cls(
            horizontal_df=loaded_well.horizontal_well,
            typewell_df=loaded_well.typewell,
            feature_columns=feature_columns,
        )
