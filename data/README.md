# Data Directory

This directory contains datasets used by the AutoMLOps pipeline.

## Credit Card Fraud Dataset

When running with `data.source: real` in `training/configs/params.yaml`, the pipeline will automatically download the Credit Card Fraud Detection dataset.

### Dataset Information

- **Source:** [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Mirror:** TensorFlow Data Server (used for automatic download)
- **Size:** ~150 MB (284,807 transactions)
- **Format:** CSV

### Features

| Feature | Description |
|---------|-------------|
| Time | Seconds elapsed from first transaction |
| V1-V28 | PCA-transformed features (anonymized) |
| Amount | Transaction amount |
| Class | Target: 1 = Fraud, 0 = Legitimate |

### Class Distribution

- **Legitimate:** 284,315 (99.83%)
- **Fraudulent:** 492 (0.17%)

### Usage

```python
from training.src.data_loader import download_credit_card_dataset, load_real_dataset

# Download dataset (if not already present)
download_credit_card_dataset()

# Load with undersampling
X, y = load_real_dataset(undersample=True, undersample_ratio=5.0)
```

### Manual Download

If automatic download fails, you can manually download:

1. Visit https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. Download `creditcard.csv`
3. Place in this directory: `data/creditcard.csv`

## Switching Between Data Sources

Edit `training/configs/params.yaml`:

```yaml
data:
  source: real       # Use Credit Card Fraud dataset
  # source: synthetic  # Use generated test data
```
