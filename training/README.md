## GlycoTrans Training Pipeline

This folder provides codes used in training **GlycoBERT** and **GlycoBART** models. The training followed the same three-step workflow:
1. Generate MS/MS Corpus
2. Prepare Tensors
3. Model Training

Training datasets are available from Zenodo [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15741423.svg)](https://doi.org/10.5281/zenodo.15741423):

- `dataset.xlsx`: Contains MS/MS data and glycan annotations
- `glycans_antennae.csv`: Contains glycan structures represented as antennae

### 1. Generate MS/MS Corpus

- Run `GenerateMSCorpus.ipynb` to generate MS/MS sentence corpus, MS/MS vocabulary, and glycan labels (for GlycoBERT) or antenna corpus (for GlycoBART).

### 2. Prepare Tensors

- Run `PrepareTensors.ipynb` to generate the tensors required for model training.
- Note that this requires the outputs from `GenerateMSCorpus.ipynb`.

### 3. Model Training

#### GlycoBERT
From command line, run:   
```python TrainGlycoBertFull.py save_dir max_seq_length num_hidden_layers num_attention_heads hidden_size batch_size num_epochs```

Training Parameters:
- ```save_dir```: directory to save the trained model
- ```max_seq_length```: maximum sequence length
- ```num_hidden_layers```: number of hidden layers
- ```num_attention_heads```: number of attention heads
- ```hidden_size```: hidden layer size
- ```batch_size```: training batch size
- ```num_epochs```: number of training epochs


#### GlycoBART
From command line, run:   
```python TrainGlycoBartFull.py save_dir num_encoder_layers num_decoder_layers num_attention_heads dim_model batch_size num_epochs```

Training Parameters: 
- ```save_dir```: directory to save the trained model
- ```num_encoder_layers```: number of encoder layers
- ```num_decoder_layers```: number of decoder layers
- ```num_attention_heads```: Number of attention heads
- ```dim_model```: dimension of layers and pooler layer
- ```batch_size```: training batch size
- ```num_epochs```: number of training epochs.
