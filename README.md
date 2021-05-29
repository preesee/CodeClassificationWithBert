# CodeClassificationWithBert

This repo provides the code for reproducing the experiments in ASTBERT: A Fine-tuned BERT model for Programming Languages based on ASTs. Unlike many other AST-based tools ASTBERT uses the subtrees of ASTs to extract syntactical information from programming languages.

For code classification, unzip and release the csv training data in folder 'data'. 
Dependency
pip install keras
pip install transformers
pip install tensflow==2.2
pip install keras-bert

run python train.py
