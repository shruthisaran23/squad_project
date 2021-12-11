## Overview

Final project for NLP CS 4650 at Georgia Tech

The goal of this project is to create a SQuAD 2.0-compatible question-answering system. The purpose is to research and implement models that will improve the performance of a baseline model known as Bidirectional Attention Flow (BiDAF) that was provided by the Stanford CS224N course. The system should be capable of reading a text and appropriately answering a question about it. A paragraph and a question about the paragraph are the system's inputs, and the system's output is the answer to the question based on the paragraph's text. During the course of this project, we modified the baseline BiDAF model to implement word-level embedding, character-level embedding and added a self-attention layer. 

We discovered that adding a character-level embedding model improves performance over the baseline. To improve the metric scores, we fine-tuned the hyper-parameters. The implementation of the character-embedding resulted in an EM score of 61.88 and an F1 score of 63.76 while integrating the self-attention layer resulted in a final EM score of 59.45 and an F1 score of 62.83.

## Setup

1. run `conda env create -f environment.yml`
    1. This creates a Conda environment called `squad`

2. Run `source activate squad`
    1. This activates the `squad` environment
  
3. Run `python setup.py`
    1. This downloads SQuAD 2.0 training and dev sets, as well as the GloVe 300-dimensional word vectors (840B)
    2. This also pre-processes the dataset for data loading

4. Run `python train.py` to train the model with hyper-parameters

5. Run `python test.py` to test the model EM and F1 scores on Dev 
