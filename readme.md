# CSE 156 PA1: Movie Review Sentiment Analyzer using a DAN with feedforwarding neural network

This repository contains a Deep Averaging Network (DAN) for sentiment analysis on movie reviews, utilizing GloVe word embeddings. The model is implemented using PyTorch and aims to classify movie reviews as positive or negative.

## How to Run

To run the different models available in the repository, follow these instructions:

1. **Part 2a - DAN (Deep Averaging Network):**
   ```bash
   python main.py --model DAN
```

2. **Part 2b - Custom Embeddings DAN:**
```bash
    python main.py --model SUBWORDDAN
```

3. **Bag of Words:**
```bash
    python main.py --model BOW
```

## Overview

The DAN model works by averaging the word embeddings in a sentence to produce a fixed-size vector representation. This vector is then passed through a feedforward neural network for sentiment classification.

## Model Architecture

- **Word Embeddings:** GloVe embeddings (`glove.6B.300d-relativized.txt`).
- **Embedding Dimension:** 300.
- **Hidden Layers:** 2 hidden layers, each with 200 dimensions.
- **Activation Function:** ReLU.
- **Output Layer:** LogSoftmax for binary classification (positive/negative).

## Training Details

- **Optimizer:** ReLU.
- **Accuracy:** The model achieves over 77% accuracy on the test set.

## Dependencies

Ensure you have the following dependencies installed (virtual environment recommended):
- Python 3.x
- PyTorch
- scikit-learn
- NumPy
- pandas

## DAN Model Overview

### SentimentDatasetDAN Class

The `SentimentDatasetDAN` class - It takes in a file containing sentiment examples and word embeddings (e.g., GloVe). The key steps in this class are:

- **Initialization (`__init__` method):**
  - Reads sentiment examples from a file.
  - Constructs sentences and labels from the examples.
  - For each word in the sentence, retrieves its corresponding embedding vector using the provided word embeddings.
  - Stacks and averages these vectors to create a single embedding for the sentence.
  - Converts all sentence embeddings and labels to PyTorch tensors.

- **Length (`__len__` method):**
  - Returns the number of examples in the dataset.

- **Item Retrieval (`__getitem__` method):**
  - Returns the sentence embedding and its corresponding label for a given index.

### NN1DAN Class

The `NN1DAN` class defines a neural network model for sentiment analysis using a Deep Averaging Network (DAN). This PyTorch `nn.Module` takes averaged sentence embeddings as input and passes them through a feedforward neural network with two hidden layers.

- **Initialization (`__init__` method):**
  - Sets up an embedding layer using the provided word embeddings (frozen for efficiency).
  - Defines two hidden layers (`fc1` and `fc2`) and an output layer (`fc3`) using linear layers.
  - Uses `ReLU` activation functions between layers and a `LogSoftmax` function for output.

- **Forward Pass (`forward` method):**
  - Applies the first hidden layer, followed by `ReLU` activation.
  - Passes the result through a second hidden layer, again using `ReLU`.
  - Finally, the output layer generates a log-probability distribution over the sentiment classes.
