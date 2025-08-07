# glycoTrans

A  repository for transformer-based models for glycan structure prediction from mass spectrometry data. Presently, the repository includes two models: **GlycoBERT** and **GlycoBART**.

(Ejas Althaf Abtheen, Arun Singh, Shyam Sriram, Changyou Chen, Sriram Neelamegham, Rudiyanto Gunawan bioRxiv 2025.07.02.662857; doi: https://doi.org/10.1101/2025.07.02.662857)

## Overview

glycoTrans introduces state-of-the-art transformer architectures to glycomics, enhancing how we predict glycan structures from tandem mass spectrometry (MS/MS) data. By treating mass spectra and glycan structures as sequences, called MS and Glycan sentence, respectively, the models are trained to capture complex contextual relationships in spectral data. 

### Key Models

- **GlycoBERT**: A BERT-based sequence classifier for high-accuracy glycan structure prediction
- **GlycoBART**: A BART-based generative model capable of de novo glycan structure inference

## Installation

### From PyPI
```bash
pip install glycotrans
```

### From GitHub
```bash
pip install git+https://github.com/CABSEL/glycotrans.git
```

## Features

###  **Superior Performance**
- **95.1% accuracy** on test datasets, outperforming state-of-the-art CNN-based methods like CandyCrunch
- Robust performance across diverse MS analysis parameters and glycan types

###  **De Novo Discovery**
- GlycoBART's generative capability enables prediction of novel glycan structures not present in training data
- Overcome database-dependent limitations

### **Transformer Architecture**
- Self-attention mechanisms capture long-range dependencies in spectral data
- Bidirectional processing using BERT and BART for comprehensive context understanding
- Custom tokenization for MS spectra and glycan structures

## Technical Innovation

### MS Sentence Representation
Our novel approach converts MS/MS spectra into "MS sentences" containing:
- Experimental metadata (LC type, ion mode, fragmentation method, etc.)
- Normalized retention time
- Precursor m/z and fragment ion information
- Peak intensity encoding through positional embeddings

### Glycan Sentence Format
Glycan structures are represented as sequences of constituent antennae:
- Terminal-to-core monosaccharide ordering
- Linkage information preservation

## Model Architecture

### GlycoBERT
- **Base**: BERT encoder with 12 transformer layers
- **Parameters**: 96 million trainable parameters
- **Task**: Multi-class classification (3,590 glycan classes)
- **Attention**: 12 attention heads per layer
- **Embedding**: 768-dimensional representations

### GlycoBART
- **Base**: BART encoder-decoder architecture
- **Parameters**: 207 million trainable parameters
- **Task**: Conditional sequence generation
- **Architecture**: 12-layer encoder + 12-layer decoder
- **Attention**: 16 attention heads per layer
- **Generation**: Beam search with 32 beams

## Performance Metrics

### Accuracy Levels
1. **Mass Accuracy**: Monoisotopic mass matching
2. **Composition Accuracy**: Monosaccharide composition matching
3. **Topological Accuracy**: Branching pattern recognition
4. **Structural Accuracy**: Complete linkage-specific identification

### Benchmark Results
| Model | Mass | Composition | Topology | Structure |
|-------|------|-------------|----------|-----------|
| GlycoBERT | 98.8% | 98.8% | 96.7% | 95.1% |
| GlycoBART (top-1) | 93.2% | 93.1% | 90.4% | 89.1% |
| GlycoBART (top-5) | 95.5% | 95.5% | 93.3% | 93.1% |
| CandyCrunch | 94.1% | 94.0% | 91.2% | 90.3% |

## Example

### `Example_Inference.ipynb` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1KEPMZ3dlQ4TeExTqsgXJ4NCeHuEp6dQn)

The Google colab notebook shows an example inference that can be modified for specific use cases.  
The example file used in the notebook can be found at [![Google Drive](https://img.shields.io/badge/Google%20Drive-4285F4?style=for-the-badge&logo=googledrive&logoColor=white)](https://drive.google.com/file/d/1Cw2sPFwBrYifYP2_U-7w3O42JvwNfypk/view?usp=share_link) 

## Data and Models

Training data are available on Zenodo under  [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15741423.svg)](https://doi.org/10.5281/zenodo.15741423).

The full version of GlycoBERT and GlycoBART are available on HuggingFace [![Sign in with Hugging Face](https://huggingface.co/datasets/huggingface/badges/resolve/main/sign-in-with-huggingface-sm.svg)](https://huggingface.co/CABSEL).

## Funding

This work was supported by funding from NHLBI (HL103411)
