# Fraud Detection Projesi - One-Class Classification

## Proje Açıklaması
Bu proje, One-Class Classification yöntemlerini kullanarak fraud (dolandırıcılık) tespiti yapmaktadır. Orijinal çalışmadan farklı olarak:

1. **Ensemble Metodlar**: Modellerin çıktılarını Voting ve Stacking ile birleştirerek daha güçlü performans elde edilmiştir.
2. **Feature Engineering**: Medicare veri setine ratio feature'lar eklenerek anomali tespiti iyileştirilmiştir.
3. **Veri Dönemi**: Bellek sınırlaması nedeniyle 2017-2019 dönemi kullanılmıştır (orijinal: 2013-2019).

### Kullanılan Modeller
- **One-Class SVM (OCSVM)** - Sigmoid ve Isotonic calibration
- **Gaussian Mixture Model (GMM)** - One-class anomaly detection
- **Isolation Forest** - Ensemble anomaly detection
- **Local Outlier Factor (LOF)** - K-NN tabanlı anomaly detection
- **OCAN** - One-Class Adversarial Networks (GAN tabanlı)

### Ensemble Metodlar
- **Soft Voting**: Tüm modellerin score ortalaması
- **Weighted Voting**: AUC performansına göre ağırlıklı ortalama
- **Stacking**: Meta-learner (Logistic Regression) ile birleştirme

### Veri Setleri
1. **Credit Card Fraud Dataset** - Kaggle'dan (~150MB)
2. **Medicare Fraud Dataset** - LEIE ve Medicare Provider verilerinin birleşimi (2017-2019, ~1.3GB)

---

## Kurulum

### 1. Gereksinimleri Yükleyin
```bash
pip install -r requirements.txt
```

### 2. Veri Setlerini Hazırlayın

#### Credit Card Dataset
Kaggle'dan indirin: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

İndirdikten sonra `creditcard.csv` dosyasını `Code_Data_Processing/` klasörüne koyun.

#### Medicare Dataset
`Combined_LEIE_Medicare_2017_2019_DOWNSIZED_1mil.csv` dosyası `data/` klasöründe mevcut.

---

## Proje Yapısı
```
proje/
├── Code_Data_Processing/          # Veri işleme notebook'ları
│   ├── credit_card_dataset_processing.ipynb
│   ├── medicare_dataset_filter_and_combining.ipynb
│   └── medicare_dataset_leie_merging.ipynb
├── Code_Model_Training/           # Model eğitim notebook'ları
│   ├── Model_Training_GMM_OCSVM_CreditCardDataset.ipynb
│   ├── Model_Training_GMM_OCSVM_MedicareDataset.ipynb
│   ├── Model_Training_IsolationForest_LOF_CreditCard.ipynb
│   ├── Model_Training_OCAN_CreditCard.ipynb
│   ├── Model_Training_OCAN_Medicare.ipynb
│   └── Model_Ensemble_Methods.ipynb   ← YENİ! Ensemble metodlar
├── Documents/                     # Rapor ve sunum dosyaları
├── outputs/                       # İşlenmiş veri setleri ve grafikler
├── pickled_storage/               # Model kayıtları ve ROC verileri
├── data/                          # Ham veri setleri
├── requirements.txt
└── README.md
```

---

## Çalıştırma Sırası

### Adım 1: Veri İşleme (İsteğe Bağlı)
Veri setlerini sıfırdan işlemek için:
```bash
# Credit Card verisini işle
jupyter notebook Code_Data_Processing/credit_card_dataset_processing.ipynb

# Medicare verisini işle
jupyter notebook Code_Data_Processing/medicare_dataset_leie_merging.ipynb
```

### Adım 2: Model Eğitimi
```bash
# Credit Card - GMM ve OCSVM
jupyter notebook Code_Model_Training/Model_Training_GMM_OCSVM_CreditCardDataset.ipynb

# Credit Card - Isolation Forest ve LOF
jupyter notebook Code_Model_Training/Model_Training_IsolationForest_LOF_CreditCard.ipynb

# Credit Card - OCAN
jupyter notebook Code_Model_Training/Model_Training_OCAN_CreditCard.ipynb

# Medicare - GMM ve OCSVM
jupyter notebook Code_Model_Training/Model_Training_GMM_OCSVM_MedicareDataset.ipynb

# Medicare - OCAN
jupyter notebook Code_Model_Training/Model_Training_OCAN_Medicare.ipynb
```

### Adım 3: Ensemble Metodlar (YENİ!)
```bash
# Ensemble karşılaştırması
jupyter notebook Code_Model_Training/Model_Ensemble_Methods.ipynb
```

---

## Sonuçları Yeniden Üretme

Tüm sonuçları otomatik olarak üretmek için:

```bash
# 1. Veri işleme
cd Code_Data_Processing
jupyter nbconvert --to notebook --execute credit_card_dataset_processing.ipynb
jupyter nbconvert --to notebook --execute medicare_dataset_leie_merging.ipynb

# 2. Model eğitimi (uzun sürebilir)
cd ../Code_Model_Training
jupyter nbconvert --to notebook --execute Model_Training_GMM_OCSVM_CreditCardDataset.ipynb
jupyter nbconvert --to notebook --execute Model_Ensemble_Methods.ipynb
```

---

## Demo için Hızlı Çalıştırma

Demo sırasında sadece Ensemble notebook'u çalıştırmanız yeterlidir:

```bash
cd Code_Model_Training
jupyter notebook Model_Ensemble_Methods.ipynb
```

Bu notebook:
- ✅ Her iki veri setini otomatik yükler
- ✅ Feature engineering uygular
- ✅ 4 farklı model eğitir
- ✅ 3 farklı ensemble yöntemi karşılaştırır
- ✅ Karşılaştırma grafiği oluşturur

---

## Veri Kaynaklarına Erişim

| Veri Seti | Bağlantı |
|-----------|----------|
| Credit Card Fraud | https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud |
| Medicare Part D | https://data.cms.gov/provider-summary-by-type-of-service/medicare-part-d-prescribers |
| LEIE Database | https://oig.hhs.gov/exclusions/exclusions_list.asp |

---

## Önemli Notlar

- GPU varsa OCAN modeli CUDA ile hızlanır
- Ensemble notebook demo için optimize edilmiştir (~5-10 dk)
- Tam veri seti eğitimi 1-2 saat sürebilir
- Credit Card dataset ~150MB, Medicare dataset ~1.3GB

---

## Proje Katkıları

### Orijinal Paper'dan Farklar:
1. **Ensemble Learning** - Modellerin birleştirilmesi
2. **Feature Engineering** - Yeni ratio feature'lar
3. **Karşılaştırmalı Analiz** - Tek model vs Ensemble performansı

### Yazar
BLG607 Data Mining - Dönem Projesi
