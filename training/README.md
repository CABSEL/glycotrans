## glycoTrans Training Pipeline

This repository provides code and instructions for training **GlycoBERT** and **GlycoBART** models for glycan analysis. Each model has its own folder (`glycobert` and `glycobart`), and both follow the same three-step workflow:

1. Generate MS Corpus
2. Prepare Tensors
3. Model Training

> ## Data Requirements

Before running any code, **download the following datasets from Zenodo** [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15741423.svg)](https://doi.org/10.5281/zenodo.15741423):

- `dataset.xlsx`
- `glycans_antennae.csv`

> ## Running the Pipeline

### 1. Generate MS Corpus

- Open `GenerateMSCorpus.ipynb` in either `glycobert` or `glycobart`.
- Run all cells in order.
- This will generate the vocabbulary, MS corpus and glycan labels or corpus needed for the next step.

### 2. Prepare Tensors

- Open `PrepareTensors.ipynb` in the same folder.
- Run all cells in order.
- Specify the locations of the vocabulary, MS corpus, glycan labels or corpus as needed.
- This will generate the tensors required for model training.

### 3. Train the Model

- Use the terminal to run the training scripts for your chosen model.
- Navigate to the directory where the TrainGlycoBertFull.py or TrainGlycoBartFull.py is saved
- Specify the output directory and other parameters as needed.
- For GlycoBERT   
```python TrainGlycoBertFull.py save_dir max_seq_length num_hidden_layers num_attention_heads hidden_size batch_size num_epochs```

- For GlycoBART  
```python TrainGlycoBartFull.py save_dir num_encoder_layers num_decoder_layers num_attention_heads dim_model batch_size num_epochs```

### Training Parameters
 
- ```save_dir``` Directory to save the trained model.
- ```max_seq_length``` Maximum sequence length.
- ```num_hidden_layers``` Number of GlycoBERT hidden layers.
- ```num_attention_heads``` Number of attention heads.
- ```hidden_size``` Hidden layer size for GlycoBERT.
- ```num_encoder_layers``` number of GlycoBART encoder layers
- ```num_decoder_layers``` number of GlycoBART decoder layers
- ```dim_model``` number of GlycoBART dimensions
- ```batch_size``` Training batch size.
- ```num_epochs``` Number of training epochs.
