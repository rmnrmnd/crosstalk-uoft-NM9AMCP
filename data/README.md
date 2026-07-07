# Dataset Directory

Place the workshop datasets in this folder. To keep the repository lightweight and follow best practices, do **not** commit large dataset files (such as `.parquet` or `.csv`) to your Git history.

## Expected Files

The training script expects the following files to be present here:
- `crosstalk_train.parquet`: The training data containing features and labels (`DELLabel`).
- `crosstalk_test_inputs.parquet`: The test data containing input features and a `RandomID` column to generate predictions.

## Setup Instructions

1. Download the dataset files from the workshop Google Drive link or Kaggle competition page.
2. Save or copy them into this folder under the names:
   ```text
   crosstalk_template/data/crosstalk_train.parquet
   crosstalk_template/data/crosstalk_test_inputs.parquet
   ```
