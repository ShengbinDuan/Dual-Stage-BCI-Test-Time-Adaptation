# Dual-Stage BCI Test-Time Adaptation

Code repository accompanying the manuscript:

**Online Adaptation via Dual-Stage Alignment and Self-Supervision for Fast-Calibration Brain-Computer Interfaces**

This work studies online test-time adaptation (TTA) for EEG-based brain-computer interfaces (BCIs), with the goal of adapting a decoder to unseen subjects without using target-domain labels. The proposed method combines distribution alignment in both the EEG input space and the intermediate representation space with a self-supervised objective for decoder updates.

The adaptation is performed sequentially on incoming target trials and supports **single-trial online updates**.

## Method

The method contains three main components.

### 1. Online Euclidean Alignment (OEA)

Euclidean Alignment is first applied in the EEG data space. During offline training, a source-domain reference covariance matrix is computed and used to align the training trials. During online testing, the reference matrix is updated as target trials arrive, and the current target trial is aligned using the updated reference.

### 2. Online Batch Normalization Statistics Update (OBN)

After data-level alignment, distribution mismatch may remain in intermediate representations. The running mean and variance of Batch Normalization layers are therefore updated online using an exponential moving average based on the current target trial.

### 3. Self-Supervised Decoder Update

Target labels are unavailable during testing. The decoder is updated using a self-supervised loss that combines:

- Shannon entropy; and
- globally smoothed pseudo-label cross-entropy.

The predicted class is used to construct a smoothed pseudo-label distribution. The validation accuracy of the pre-trained source decoder is used as the global smoothing parameter and is fixed for all target trials of the held-out test subject.

## Online Adaptation Procedure

For each incoming target trial, the procedure follows this order:

1. update the Euclidean Alignment reference matrix;
2. align the current EEG trial;
3. update Batch Normalization statistics;
4. compute the predictive distribution and obtain the current prediction;
5. record the prediction for evaluation;
6. construct the globally smoothed pseudo-label distribution;
7. compute the self-supervised loss; and
8. update the decoder parameters for subsequent trials.

The parameter update is performed after the prediction of the current trial, and the updated parameters are used for the next trial. No target-domain ground-truth labels are used for test-time adaptation.

## Experimental Evaluation

Experiments follow a **leave-one-subject-out (LOSO)** protocol. In each fold, one subject is held out for testing and the remaining subjects form the training fold. Within the training fold, trials from each subject are split chronologically: the first 80% are used for training and the last 20% are used for validation. The held-out test subject is not used for offline training, validation, or hyperparameter selection.

The method is evaluated on two EEG paradigms, five public datasets, and seven decoder backbones.

### SSVEP

| Dataset | Subjects used in the experiments | Backbones |
| --- | ---: | --- |
| Benchmark | 35 | CNN-M, CNN-C, FBCNN-M, FBCNN-C |
| 12JFPM | 10 | CNN-M, CNN-C, FBCNN-M, FBCNN-C |

For SSVEP, trials are segmented into non-overlapping 1-second windows for both training and testing, following the settings used by the corresponding baseline studies.

### Motor Imagery

| Dataset | Subjects used in the experiments | Backbones |
| --- | ---: | --- |
| PhysioNet | 105 | EEGNet, DeepConvNet, EEGConformer |
| Moritz | 10 | EEGNet, DeepConvNet, EEGConformer |
| Zhou2016 | 4 | EEGNet, DeepConvNet, EEGConformer |

For motor imagery, the full trial length is used. The PhysioNet experiments use 105 subjects after excluding four subjects from the original cohort because of data-quality issues reported in the manuscript.

## Reported Results

Relative to the corresponding source-only baselines, the proposed method reports average accuracy gains of:

- **4.90 percentage points** on the SSVEP experiments; and
- **3.64 percentage points** on the motor-imagery experiments.

The dataset-level average accuracy gains reported in the manuscript are:

| Dataset | Average accuracy gain (percentage points) |
| --- | ---: |
| Benchmark | +7.58 |
| 12JFPM | +2.23 |
| PhysioNet | +1.61 |
| Moritz | +3.47 |
| Zhou2016 | +5.84 |

The method is compared with representative test-time adaptation approaches including **AdaBN**, **Tent**, **T3A**, and **T-TIME**. The manuscript also reports ablation studies for Online Euclidean Alignment, online Batch Normalization statistics updates, and the self-supervised loss.

## Reference Experimental Settings

The manuscript reports the following implementation and adaptation settings:

| Setting | Value |
| --- | --- |
| Framework | PyTorch 1.13.0 |
| GPU used in the reported experiments | NVIDIA Tesla V100 |
| OEA weight, `ω` | 500 |
| BN EMA weight, `α` | 0.7 |
| Numerical stability term, `ε` | `3 × 10^-5` |
| Self-supervised loss weight, `λ` | 1.1 |

Other dependency versions are not specified in the manuscript and are therefore not listed here.

## Repository Contents

The current repository contains:

```text
.
├── README.md
├── TTA.ipynb
└── TTA_CNN_M.py
```

`TTA.ipynb` and `TTA_CNN_M.py` are the code files currently provided in this repository. This README does not prescribe command-line arguments, dataset paths, or additional dependencies that are not explicitly documented in the manuscript or visible repository contents.

## Reproducibility Notes

Several details are important when reproducing the reported protocol:

- the held-out test subject must remain excluded from offline training, validation, and hyperparameter selection;
- validation accuracy is computed from the validation portion of the training fold and is used as the global pseudo-label smoothing parameter;
- adaptation proceeds sequentially over target trials;
- the prediction for the current trial is recorded before the decoder parameter update derived from that trial; and
- no target-domain ground-truth labels are used during adaptation.

## Limitations

The manuscript notes that, although mean accuracy improves across the evaluated datasets, some baseline comparisons do not reach statistical significance. The relatively small subject cohorts in 12JFPM and Moritz (10 subjects each) and Zhou2016 (4 subjects) may limit statistical power. Evaluation on larger subject cohorts is identified as an important direction for assessing the consistency of the improvements across datasets and decoder backbones.

The manuscript also identifies real-world deployment of the proposed online adaptation algorithm in practical BCI systems as a direction for future evaluation.

## Paper

**Online Adaptation via Dual-Stage Alignment and Self-Supervision for Fast-Calibration Brain-Computer Interfaces**

Citation metadata is not included here because the provided manuscript is anonymized.
