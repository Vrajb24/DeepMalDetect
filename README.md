
# Android Malware Detection and Classification

This repository contains code for detecting and classifying Android applications as malware or benign based on system call sequences. The project utilizes several approaches: Graph Neural Networks, N-Grams with Random Forest, Recurrent Neural Networks (RNNs), and Transformer models. Each approach leverages unique data preprocessing and feature extraction techniques to improve classification accuracy.

## Table of Contents
- Project Overview
- Dataset
- Folder Structure
- Installation
- Usage
- Approaches
  - Graph Neural Network Models
  - N-Gram Model
  - Simple Encoding with RNN
  - Transformer Model
- Results and Evaluation
- Contributors
- License

## Project Overview

This project aims to classify Android applications using system call sequences to detect malware. The "Maldroid 2020" dataset is used, containing various types of malware and benign applications. Through multiple approaches, we explore the efficacy of different machine learning models for this classification task.

## Dataset

The Maldroid 2020 dataset is utilized, consisting of:
- Raw APK Files (Android application files)
- JSON Files (containing static and dynamic analysis data)
- CSV Files (with frequency data summaries)

The JSON files, providing detailed system call data, are the primary data source.

## Folder Structure

Each folder represents a distinct machine learning approach:

### Graph_Model
Contains two variants of Graph Neural Network models.

- **Model 1**
  - `extract_procname_sysname.ipynb`: Extracts unique system calls from JSON data.
  - `graphify.ipynb`: Converts JSON data into graph data based on system calls.
  - `labeler.ipynb`: Labels the graph data for supervised learning.
  - `GraphModel1.ipynb`: Defines and trains the first variant of the Graph Neural Network model.

- **Model 2**
  - `extract_procname_sysname.ipynb`: Similar to Model 1, extracts unique system calls.
  - `graphify2.ipynb`: Converts and labels JSON data for the second graph-based model.
  - `GraphModel2.ipynb`: Defines and trains the second variant of the Graph Neural Network model.

### N_Gram_Model
Contains scripts for feature extraction and model training using an N-Gram approach.

- `FeatureExtractor.ipynb`: Extracts features, both dynamic and static, from JSON data.
- `NGram_model.ipynb`: Implements a Random Forest model using extracted N-Gram features.

### Simple_Encoding
Contains files related to an RNN model utilizing simple encoding.

- `dataextractor.ipynb`: Extracts and encodes system call sequences from JSON data.
- `Simplemodel.ipynb`: Defines and trains an RNN model on the extracted and encoded data.

### Transformer_Model
Contains preprocessing scripts and Transformer model code.

- `TxtSeqExtractor.ipynb`: Extracts system call sequences in text format.
- `Transformer_model.ipynb`: Creates embeddings from the text data and trains a Transformer model for classification.

## Installation

To run the code, ensure you have Python installed. Install the required packages using:

```bash
pip install -r requirements.txt
```
The `requirements.txt` file should list all dependencies used across different notebooks.

## Usage

Each model has its own preprocessing and training scripts. Follow these steps for each approach:

1. **Data Preprocessing**: Run the respective data extraction and preprocessing notebooks to prepare the data for the model.
2. **Model Training**: Execute the model script in each folder to train the model on preprocessed data.

## Approaches

### Graph Neural Network Models
- **Description**: Utilizes Graph Neural Networks to model interactions in system call sequences.
- **Preprocessing**: Converts JSON data into graph format with nodes representing system calls and edges showing their relationships.
- **Model Variants**: Two variants are provided in `GraphModel1.ipynb` and `GraphModel2.ipynb`.

### N-Gram Model
- **Description**: Employs an N-Gram approach to extract local patterns in system call sequences, with a Random Forest classifier.
- **Preprocessing**: Feature extraction includes both dynamic (sequence-based) and static features.
- **Model Training**: Defined in `NGram_model.ipynb`.

### Simple Encoding with RNN
- **Description**: Uses simple encoding of system call sequences with an RNN model.
- **Preprocessing**: Sequences are extracted and encoded as numeric data, suitable for sequential learning.
- **Model Training**: Defined in `Simplemodel.ipynb`.

### Transformer Model
- **Description**: Leverages a Transformer model for context-based learning on system call sequences.
- **Preprocessing**: Converts system calls to a text-based sequence, then embeds each sequence for model training.
- **Model Training**: Defined in `Transformer_model.ipynb`.

## Results and Evaluation

Each approach is evaluated using standard metrics such as accuracy, precision, recall, and F1 score. Results indicate that the Transformer and N-Gram models perform best in capturing complex patterns and local sequence information, respectively.

## Contributors

- Vraj Patel
- Pavani Priya
- Akshat Shrivastav
- Divya Krupa

## License

This project is licensed under the MIT License.
