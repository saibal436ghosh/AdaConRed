# AdaConRed
Official implementation accompanying the paper "Adaptive Conformal Redistribution for Inter-class Transitional Uncertainty in Medical Image Classification."

## Overview

AdaConRed is a post-conformal decision-refinement framework for
uncertainty-aware medical image classification in the presence of
between-class transitional ambiguity.

The repository contains implementations of:

1. Feature extraction using a frozen vision foundation model followed by
   lightweight MLP classification.
2. Standard split conformal prediction using the LAC nonconformity score.
3. Adaptive conformal prediction using the proposed entropy-modulated,
   margin-aware nonconformity score.
4. Label-free transitional redistribution based on the resulting
   prediction sets.

## Repository Structure

```text
AdaConRed/
├── README.md
├── requirements.txt
├── .gitignore
└── src/
    ├── encoder_classifier.py
    ├── split_conformal_red.py
    └── adaptive_conformal_red.py
