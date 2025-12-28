# Fraud Detection Project - One-Class Classification

## Project Description

This project implements fraud detection using One-Class Classification methods. In contrast to the original study, the following enhancements were made:

1. **Ensemble Methods**: Model outputs were combined using Voting and Stacking techniques to achieve stronger performance.
2. **Feature Engineering**: Ratio features were added to the Medicare dataset to improve anomaly detection.
3. **Data Period**: Due to memory constraints, the 2017-2019 period was used (original: 2013-2019).

### Models Used
- **One-Class SVM (OCSVM)** - Sigmoid and Isotonic calibration
- **Gaussian Mixture Model (GMM)** - One-class anomaly detection
- **Isolation Forest** - Ensemble anomaly detection
- **Local Outlier Factor (LOF)** - K-NN based anomaly detection
- **OCAN** - One-Class Adversarial Networks (GAN-based)

### Ensemble Methods
- **Soft Voting**: Average of all model scores
- **Weighted Voting**: Weighted average based on AUC performance
- **Stacking**: Combination using meta-learner (Logistic Regression)

### Datasets
1. **Credit Card Fraud Dataset** - From Kaggle (~150MB)
2. **Medicare Fraud Dataset** - Combination of LEIE and Medicare Provider data (2017-2019, ~1.3GB)

---

## Installation

### 1. Install Requirements
```bash
pip install -r requirements.txt
```

### 2. Prepare Datasets

#### Credit Card Dataset
Download from Kaggle: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

After downloading, place the `creditcard.csv` file in the `Code_Data_Processing/` folder.

#### Medicare Dataset
The `Combined_LEIE_Medicare_2017_2019_DOWNSIZED_1mil.csv` file is available in the `data/` folder.

---

## Project Structure
```
project/
├── Code_Data_Processing/          # Data processing notebooks
│   ├── credit_card_dataset_processing.ipynb
│   ├── medicare_dataset_filter_and_combining.ipynb
│   └── medicare_dataset_leie_merging.ipynb
├── Code_Model_Training/           # Model training notebooks
│   ├── Model_Training_GMM_OCSVM_CreditCardDataset.ipynb
│   ├── Model_Training_GMM_OCSVM_MedicareDataset.ipynb
│   ├── Model_Training_IsolationForest_LOF_CreditCard.ipynb
│   ├── Model_Training_OCAN_CreditCard.ipynb
│   ├── Model_Training_OCAN_Medicare.ipynb
│   └── Model_Ensemble_Methods.ipynb   # NEW - Ensemble methods
├── Documents/                     # Report and presentation files
├── outputs/                       # Processed datasets and plots
├── pickled_storage/               # Model checkpoints and ROC data
├── data/                          # Raw datasets
├── requirements.txt
└── README.md
```

---

## Execution Order

### Step 1: Data Processing (Optional)
To process datasets from scratch:
```bash
# Process Credit Card data
jupyter notebook Code_Data_Processing/credit_card_dataset_processing.ipynb

# Process Medicare data
jupyter notebook Code_Data_Processing/medicare_dataset_leie_merging.ipynb
```

### Step 2: Model Training
```bash
# Credit Card - GMM and OCSVM
jupyter notebook Code_Model_Training/Model_Training_GMM_OCSVM_CreditCardDataset.ipynb

# Credit Card - Isolation Forest and LOF
jupyter notebook Code_Model_Training/Model_Training_IsolationForest_LOF_CreditCard.ipynb

# Credit Card - OCAN
jupyter notebook Code_Model_Training/Model_Training_OCAN_CreditCard.ipynb

# Medicare - GMM and OCSVM
jupyter notebook Code_Model_Training/Model_Training_GMM_OCSVM_MedicareDataset.ipynb

# Medicare - OCAN
jupyter notebook Code_Model_Training/Model_Training_OCAN_Medicare.ipynb
```

### Step 3: Ensemble Methods (NEW)
```bash
# Ensemble comparison
jupyter notebook Code_Model_Training/Model_Ensemble_Methods.ipynb
```

---

## Reproducing Results

To automatically reproduce all results:

```bash
# 1. Data processing
cd Code_Data_Processing
jupyter nbconvert --to notebook --execute credit_card_dataset_processing.ipynb
jupyter nbconvert --to notebook --execute medicare_dataset_leie_merging.ipynb

# 2. Model training (may take a long time)
cd ../Code_Model_Training
jupyter nbconvert --to notebook --execute Model_Training_GMM_OCSVM_CreditCardDataset.ipynb
jupyter nbconvert --to notebook --execute Model_Ensemble_Methods.ipynb
```

---

## Quick Demo Execution

For the demo, it is sufficient to run only the Ensemble notebook:

```bash
cd Code_Model_Training
jupyter notebook Model_Ensemble_Methods.ipynb
```

This notebook:
- Automatically loads both datasets
- Applies feature engineering
- Trains 4 different models
- Compares 3 different ensemble methods
- Generates comparison plots

---

## Data Source Access

| Dataset | Link |
|---------|------|
| Credit Card Fraud | https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud |
| Medicare Part D | https://data.cms.gov/provider-summary-by-type-of-service/medicare-part-d-prescribers |
| LEIE Database | https://oig.hhs.gov/exclusions/exclusions_list.asp |

---

## Important Notes

- If GPU is available, the OCAN model is accelerated with CUDA
- The Ensemble notebook is optimized for demo (~5-10 min)
- Full dataset training may take 1-2 hours
- Credit Card dataset ~150MB, Medicare dataset ~1.3GB

---

## Project Contributions

### Differences from the Original Paper:
1. **Ensemble Learning** - Combination of models
2. **Feature Engineering** - New ratio features
3. **Comparative Analysis** - Single model vs Ensemble performance

### Author
**Bahakirbasogluu** - BLG607 Data Mining Term Project

GitHub Repository: https://github.com/Bahakirbasogluu/Fraud-Detection
