## Model Writeup

We used all fingerprints available in the dataset to train 11 base models. 9 models are trained on
the single fingerprints, while the two remaining models use a mixture of ECFP4+TOPTOR and
FCFP4+AVALON. In our experiments, the ensemble of these 11 models produced the best results.

The data is split into training and validation (80 and 20%). The split was stratified by the first building block 
(BB1) in the DEL_ID. I tried splitting by the DEL library number, but the results were not stable
for further development. The 80% training portion comes from
four of the five folds, and the held-out fold (20%) is the validation set.

*Base models architecture:* the base models' architecture is inspired by wide-and-deep deep-learning
models. The input vector has 4 paths to reach the output. First, it is connected to the output
(logits) with a single linear layer. The second path from input to output has a single 1024-neuron
hidden layer. The third path has a 1024-neuron and a 512-neuron hidden layer. And the fourth path has
a 1024-, 512-, and 256-neuron hidden layer. It is trained with a class-weighted binary cross-entropy
(BCE) loss (a positive-class weight handles the ~12:1 label imbalance) and the AdamW optimizer.
After training the 11 base models one by one, we used the outputs of these models to train the
ensembler on the validation dataset (to prevent data leakage in training the ensembler).

*Metrics Used:* for selecting the best epoch and saving the best model, we used AUPRC on the
validation set. For smoke-testing the models, we monitored precision at 200 (P@200).

*Ensembler:* after training the base models, we trained a set of ensemblers to find the consensus
between the 11 base models, and then we mixed them to generate the submission. The ensemblers are as
follows:

1. **TabPFN-v3:** we feed a class-balanced subsample of ~10,000 validation molecules (about 13% of the
   validation fold), represented by their probabilities from the 11 base models, as the in-context
   training set for TabPFN, and use it to score the remaining molecules and the test set.
2. **Rank-average:** we rank-normalize each base model's predictions to [0, 1] and average the ranks
   across the 11 models.
3. **Rank-median:** the same rank-normalization, but we take the median rank across the 11 models
   instead of the mean, which is more robust to any single base model being off.

We combine the three ensemblers by rank-averaging their outputs, and this blended ranking is the final
submission. We also tried an MLP meta-learner as a fourth ensembler, but it had the highest
validation AUPRC yet the worst ranked-metric score, so it was dropped.

---

## Structure

```text
submission_template/
├── README.md             <- This file (contains your writeup)
├── requirements.txt      <- Package requirements
├── train_model.py        <- Training script (saves your model and generates predictions)
├── data/
│   └── README.md         <- Dataset setup instructions
├── models/
│   └── best_model.pkl    <- Your final trained model (save it here)
└── src/
    ├── __init__.py
    ├── dataset.py        <- Data loaders
    └── eval.py           <- Evaluation functions
```

---

## How to Run

1. **Install UV**: to reproduce the results please install uv. pip will not work correctly.
1. **Downlaod PKL file**: the .pkl file is 1 GB. I uploaded it to "https://drive.google.com/file/d/1NNu-0dLMNv9QPGSTD2vP5erBAAmQNGQO/view?usp=sharing". You can download and place it in model/. 
1. **Install packages**:
   ```bash
   uv sync
   ```
1. **Place datasets** in `data/` (see `data/README.md` for download links).
1. **Run the script**:
   ```bash
   uv run python train_model.py
   ```
   This script will run your local validation with confidence intervals, train the final model on all data, save it to `models/best_model.pkl`, and output `submission.csv`.
