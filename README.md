## PII Data Detection & Removal from Educational Data 

This repository contains the full pipeline and solution for the Kaggle competition **PII Data Detection**, which involves identifying and removing personally identifiable information (PII) from unstructured student essays using advanced NLP techniques. This is a **Named Entity Recognition (NER)** task, framed as a **token classification** problem. 

📅 Competition Link: [Kaggle PII Data Detection](https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data)


## Quick start

You can close the repository, setup an environment, and start with the scripts.

## Environment Setup 

Let's first clone the repository:
```bash
git clone https://github.com/deepakpokkalla/pii-data-detection.git
cd pii-data-detection
```

Using your preferred environment manager, create an environment and simply install these packages:

If you want to use `miniconda`:
```bash
conda create --name pii python=3.12
conda activate pii
```

Simply install the packages from the `requirements.txt`

```bash
pip install -r requirements.txt
```

**Dependencies**: 

Hugging Face Libraries: 
- `transformers`: Pretrained Transformer models 
- `accelerate`: Simplifies distributed training and inference CPU, GPU, TPU
- `datasets`: Easy access and processing of standard NLP datasets
- `peft`: Parameter-Efficient Fine-Tuning methods for large language models
- `trl`: Utilities for Reinforcement Learning with Transformers
- `bitsandbytes`: Memory-efficient 8-bit optimizers for large models
- `sentence-transformers`: Sentence embeddings for semantic search and clustering 
- `sentencepiece`: Subword tokenizer used by many transformer models

Other Libraries:
- `torch` 
- `numpy` 
- `pandas`
- `einops`: Flexible tensor operations (rearranging, reshaping, repeating)
- `hydra-core`: Configuration management for complex projects.
- `pynvml`: NVIDIA Management Library Python bindings for GPU monitoring.
- `wandb` for logging

## Datasets

Let's install kaggle package and setup authentication before downloading the dataset. 

Simply install the kaggle package: 
```bash
pip install kaggle 
```

Setup the API token following the instructions provided [here](https://www.kaggle.com/docs/api).

Let's download the dataset from kaggle and setup:

```bash
# create a datasets folder
mkdir datasets
cd datasets

# download and unzip
kaggle competitions download -c pii-detection-removal-from-educational-data
unzip pii-detection-removal-from-educational-data.zip -d pii-detection-removal-from-educational-data
rm pii-detection-removal-from-educational-data.zip

# rename the original kaggle data folder using powershell or cmd
# using powershell 
Rename-Item -Path ".\pii-detection-removal-from-educational-data" -NewName "pii-kaggle-data"
# using cmd
rename "pii-detection-removal-from-educational-data" pii-kaggle-data
```

## Training

To train a PII detection model, simply use the provided training script of corresponding model. 

```bash 
python train_d_alpha.py
```
which will use the `config/d_alpha/conf_d_alpha.yaml`

## Citation

If you like the project and want to use it somewhere, please use this citation: 

```
@misc{pokkalla2024piidetection,
  author = {Deepak Pokkalla and Raja Biswas},
  title = {pii-data-detection},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/deepakpokkalla/pii-data-detection.git}}
}
```
