# Global TVT Feature Engineering

Bu dokuman mevcut global TVT baseline'inin kullandigi feature setini aciklar.
Model PS'ye gore delta TVT degil, her satirdaki mutlak TVT degerini ogrenir.

Kaynaklar: [Kaggle veri sayfasi](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/data) ve [gorev sunumu metin cikarimi](https://github.com/vamseeachanta/kaggle-rogii-2026/blob/main/docs/task-brief.md).

---

## 1. Model Sozlesmesi

~~~text
X = MD, X, Y, Z, GR, trajectory ve GR-pattern feature'lari
y = TVT
~~~

| Konu | Mevcut davranis |
|------|-----------------|
| Train satirlari | Her train kuyusunun tum satirlari |
| Train hedefi | Mutlak TVT |
| Model feature'i | 29 sayisal feature |
| TVT | Yalniz hedef; asla X icinde degil |
| TVT_input | Feature olarak kullanilmaz |
| PS | Model girdisi degil |
| Validation skoru | Yalniz gercek PS sonrasi satirlarda RMSE |
| Submission | Yalniz gercek PS sonrasi satirlar yazilir |

PS, bu model icin jeolojik bilgi kaynagi degildir. Yarismada hangi satirlarin
maskelendigini ve skorlandigini belirleyen bir sinirdir. Train'de tum jeolojik
gecisler ogrenilir; validation ve submission yarismayla ayni maskede degerlendirilir.

---

## 2. Veri Kavramlari

| Alan | Anlam |
|------|-------|
| MD | Measured Depth. Kuyu boyunca gidilen mesafe; kuyu egriyse dusey konuma esit degildir. |
| X, Y, Z | Kuyu trajectory koordinatlari. Z dusey trajectory koordinatidir; TVT ile ayni degildir. |
| GR | Gamma Ray logu. Kaya tipinin veya litolojinin imzasini tasir; bazi satirlarda NaN olabilir. |
| TVT | Tahmin hedefi olan True Vertical Thickness referans degeri. Horizontal satirin typewell TVT eksenindeki jeolojik konumunu temsil eder. |
| TVT_input | Yarismada PS'ye kadar verilen TVT prefix'i. Mevcut global baseline bunu kullanmaz. |

Standart jeolojik tanimda TVT, sabit bir yatay konumda katmanin upper ve lower
boundary'leri arasindaki dikey mesafedir. Yarismadaki TVT kolonu ise satir
boyunca degisen ve typewell GR profiliyle hizalanan bir jeolojik referans ekseni
olarak kullanilir; tek bir satirdaki katmanin toplam kalinligi gibi okunmaz.

~~~text
MD  : kuyu boyunca mesafe
Z   : kuyunun fiziksel dusey koordinati
TVT : jeolojik TVT referansindaki konum
~~~

Dikey bir kuyuda ayni katmanin iki siniri icin MD_bottom - MD_top, TVT'ye
yaklasabilir. Tek bir MD degeri TVT degildir. Typewell dosyasinda MD kolonu
bulunmaz; referans ekseni TVT'dir.

---

## 3. Aktif Feature Seti

"training.py" icindeki "BASELINE_FEATURE_COLUMNS" toplam 29 feature tanimlar.

### 3.1 Ham olcumler (5)

~~~text
MD, X, Y, Z, GR
~~~

Bu kolonlar modelin kuyu konumunu ve yerel log olcumunu dogrudan gormesini
saglar. GR NaN olabilir; tree tabanli modeller eksik degerleri isleyebilir.
Missing feature'lari da bu sinyalin guvenilirligini ayri olarak verir.

### 3.2 Trajectory ve geometri (12)

Bu feature'lar "add_geometry_features()" tarafindan uretilir.

| Feature | Formul | Ne anlatir? |
|---------|--------|--------------|
| dMD | MD[t] - MD[t-1] | Ardisik olcumler arasindaki kuyu boyunca mesafe. |
| dX, dY, dZ | Koordinat farki | Kuyunun yerel hareket vektoru. |
| dx_per_md, dy_per_md, dz_per_md | dX/dMD, dY/dMD, dZ/dMD | Bir feet MD basina yon degisimi. |
| horizontal_step | sqrt(dX^2 + dY^2) | Yerel yatay ilerleme. |
| inclination_deg | acos(abs(dz_per_md)) | Kuyunun yataya gore egimi. 90 derece yatay, 0 derece dikeye yakindir. |
| azimuth_sin, azimuth_cos | sin/cos(arctan2(dY, dX)) | Yatay sondaj yonu; aci sarimini onler. |
| dogleg_angle_deg | Ardisik hareket vektorlerinin acisi | Kuyunun ne kadar sert yon degistirdigi. |

Jeolojik katmanlar duz, egimli veya lokal olarak degisken olabilir. Bu nedenle
yalniz Z degil, kuyunun yatay ilerleme yonu ve yerel egimi de TVT icin sinyal
tasir.

Ilk satirda fark tabanli feature'lar NaN olur. LightGBM ve CatBoost bu degeri
dogrudan isleyebilir.

### 3.3 GR eksiklik yapisi (4)

Bu feature'lar "add_gr_missing_features()" tarafindan uretilir.

| Feature | Anlam |
|---------|-------|
| GR_isna | O satirdaki GR degerinin eksik olup olmadigi. |
| GR_missing_run_length | Satirin bulundugu ardisik GR-NaN blogunun uzunlugu. |
| steps_since_valid_GR | Son gecerli GR olcumunden beri gecen satir sayisi. |
| steps_until_valid_GR | Sonraki gecerli GR olcumune kadar satir sayisi. |

GR eksikligi rastgele olmayabilir. Model, GR sinyalinin bulunmadigi uzun bir
aralik ile tek bir eksik olcumu ayni durum olarak yorumlamaz.

"GR_interpolated" fonksiyon icinde lag/lead deneyleri icin uretilir, fakat
mevcut 29-feature baseline listesine dahil degildir.

### 3.4 Yerel GR deseni (8)

"add_gr_rolling_features()" GR'nin iki farkli olcekteki yerel imzasini uretir.

| Pencere | Feature'lar | Ne anlatir? |
|---------|-------------|-------------|
| 10 satir | GR_roll_mean_10, GR_roll_std_10, GR_roll_range_10, GR_missing_fraction_10 | Kisa mesafedeki GR seviyesi, oynakligi, araligi ve veri kalitesi. |
| 50 satir | GR_roll_mean_50, GR_roll_std_50, GR_roll_range_50, GR_missing_fraction_50 | Daha genis jeolojik veya log imzasi. |

Pencereler "center=True" ile hesaplanir. Bu testte tum GR serisi dosyada
bastan bulundugu icin yasaldir: gelecekteki TVT veya TVT_input kullanilmaz.

---

## 4. Aktif Pipeline

"make_baseline_feature_functions()" su sirayla calisir:

~~~text
set_row_idx
  -> add_geometry_features
  -> add_gr_missing_features
  -> add_gr_rolling_features(window_sizes=(10, 50), center=True)
~~~

"set_row_idx" MD siralamasini stabil hale getirir ve dahili satir indeksi
ekler. "row_index" model feature'i degildir.

Bu pipeline'da asagidakiler cagrilmaz:

~~~text
add_ps_position_features
add_md_features
add_spatial_ps_features
add_ps_tvt_features
add_typewell_features
~~~

Dolayisiyla aktif modelde ps_idx, steps_after_ps, md_after_ps,
last_known_TVT, PS'ye gore XYZ farklari veya typewell ozet feature'lari yoktur.

---

## 5. Egitim ve Validation

"build_training_dataset()" iki satir secim modu destekler.

~~~python
# Model tum jeolojik gecisleri gorur.
X_train, y_train, train_groups = build_training_dataset(
    train_well_ids,
    BASELINE_FEATURE_COLUMNS,
    make_baseline_feature_functions,
    post_ps_only=False,
)

# Kaggle metrigini taklit eder.
X_valid, y_valid, valid_groups = build_training_dataset(
    valid_well_ids,
    BASELINE_FEATURE_COLUMNS,
    make_baseline_feature_functions,
    post_ps_only=True,
)
~~~

Mevcut notebook rastgele group split yerine sabit bir holdout kullanir: Kaggle
testindeki uc well ID train listesinden tamamen cikarilir. Bu ID'lerin train
kopyalari gercek TVT etiketini saglar; ortak input kolonlari test CSV'leriyle
birebir aynidir. Bu nedenle model bu uc kuyuyu egitimde hic gormeden test-like
bir RMSE verir.

~~~text
RMSE = sqrt(mean((TVT_true - TVT_pred)^2))
~~~

Yalniz PS sonrasi holdout satirlari skora girer. Modelin mutlak TVT ogrenmesi
ile yarismadaki degerlendirme siniri birbirinden ayrilmistir.

---

## 6. Leakage Kurallari

| Veri | Aktif global modelde durum |
|------|----------------------------|
| Train TVT | Sadece y_train; feature degil. |
| Test TVT | Dosyada yok; tahmin edilen deger. |
| TVT_input | Kullanilmaz. |
| Train-only formasyon kolonlari | Kullanilmaz; testte yoktur. |
| Tum GR serisi | Ham, missing ve centered rolling feature'lari icin kullanilir. |
| Typewell TVT/GR | Bu baseline'da kullanilmaz. |

Bu kurallar sayesinde train ve test feature semasi aynidir. Train tarafindaki
gercek TVT, modelin kendisini yeniden gormesine izin verilmeden sadece hedef
olarak kullanilir.

---

## 7. Typewell ve PS-Aware Kodun Durumu

"feature_engineering.py" ve "typewell_alignment.py" icinde daha once denenmis
PS-aware ve typewell fonksiyonlari korunmaktadir. Bunlar mevcut global
baseline'in parcasi degildir.

| Deney | Neden aktif degil? |
|-------|--------------------|
| PS'ye gore delta TVT | Modeli verilen prefix'e bagliyordu; global model mutlak TVT ogreniyor. |
| PS'ye gore mesafe ve XYZ farklari | PS referansi gerektirir. |
| PS-oncesi TVT trend feature'lari | TVT_input kullanir. |
| Kuyu-seviyesi typewell ozetleri | Deneyde belirgin fayda saglamadi; cogu satirda sabitti. |
| Row-level typewell refinement | Delta TVT ve last_known_TVT gerektirir; sequence-aware degildir. |

Typewell sonraki model ailesi icin yine de degerlidir. Dogru sonraki adim,
horizontal ve typewell GR dizilerini tekil GR degeriyle degil, ardisik GR
deseniyle eslestiren sequence-aware bir model kurmaktir. Bu model TVT'nin
artmasina, azalmasina ve sabit kalmasina izin vermelidir.

---

## 8. Ilgili Kod

| Dosya | Sorumluluk |
|-------|------------|
| feature_engineering.py | Ham feature fonksiyonlari. |
| training.py | Aktif 29-feature listesi ve coklu-kuyu dataset builder. |
| training_data.py | Bir kuyunun feature ve target secimi. |
| baseline_training.ipynb | LightGBM, HistGradientBoosting ve CatBoost deneyleri. |
| typewell_alignment.py | Aktif olmayan PS-aware row-level typewell deneyi. |

Mevcut ana model, mutlak TVT hedefiyle egitilen "CatBoostRegressor" veya
"LGBMRegressor"dur.

