# Loss Landscape for Active Learning 2025

## Overview

This project provides a framework for analyzing neural network loss landscapes through Hessian eigenvector computation and 2D planar interpolation, specifically designed for materials science applications using ALIGNN models. The toolkit enables unsupervised machine learning analysis of loss landscape patterns to inform active learning strategies and dataset pruning decisions.

## Installation

```bash
git clone https://github.com/your-username/Loss_landscape_for_active_learning_2025.git
cd Loss_landscape_for_active_learning_2025
```

## Environment Setup

Create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate losslandscape-ACSURP
```

## How to use

### 📚 Demo Notebooks
*Strongly recommended for new users - follow sequentially to understand key functionality:**

1. **`demo/demo1_loss_landscape.ipynb`** - Complete tutorial covering Hessian eigenvector computation and 2D loss landscape generation
2. **`demo/demo2_loss_landscape_automated.ipynb`** - Automated workflow for loss landscape analysis
3. **`demo/demo3_post_processing.ipynb`** - Post-processing, feature extraction, and landscape metrics computation
4. **`demo/demo4_ML_analysis.ipynb`** - Machine learning analysis including PCA, clustering, and visualization

### 📊 Analysis Notebooks

#### 1. Evolution of Active Learning Models
**Start here to understand model progression across active learning iterations:**
- **`NBE5_Analyzing_evolution_of_alignn_model.ipynb`** - **⭐ START HERE** - Comprehensive analysis of how ALIGNN models evolve across active learning iterations
- **`NBE2_Analyzing_iter_1_alignn_model.ipynb`** - Detailed analysis of iteration 1 model
- **`NBE3_Analyzing_iter_2_alignn_model.ipynb`** - Detailed analysis of iteration 2 model  
- **`NBE4_Analyzing_iter_3_alignn_model.ipynb`** - Detailed analysis of iteration 3 model

#### 2. Analysis of JARVIS Pre-trained Models
**Loss landscape analysis of production-ready materials science models:**
- **`NBE7_Analyzing_JVDFT_dHf_Loss_landscapes.ipynb`** - Formation energy (dHf) prediction model analysis
- **`NBE8_Analyzing_JVDFT_bandgap_Loss_landscapes.ipynb`** - Bandgap prediction model analysis

#### 3. Loss Landscape-Based Dataset Pruning
**Intelligent dataset reduction using loss landscape insights:**
- **`NBE9_Creating_LL_pruned_dataset.ipynb`** - Creating loss landscape-based pruned datasets
- **`NBE10_Analyzing_LL_pruning.ipynb`** - Comparative analysis of LL pruning vs. random sampling and Query by Committee

## Key Features

- **Hessian Eigenvector Computation**: Efficient calculation of maximum and minimum eigenvectors of loss Hessian matrices
- **2D Loss Landscape Generation**: Planar interpolation between original and eigenvector models
- **Unsupervised Analysis**: PCA, spectral clustering, and pattern recognition in loss landscape space

## Getting Started

- **New Users**: Start with the demo notebooks (`demo1` → `demo2` → `demo3` → `demo4`)


## Citation

- Using a custom version of [`loss_landscapes`](https://github.com/marcellodebernardi/loss-landscapes/tree/master) by Marcello de Bernardi.
- Using functions written by Dr. Ashley Dale for Hessian eigenvector computation.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.