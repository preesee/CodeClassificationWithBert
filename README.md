# CodeClassificationWithBert

This repo provides the code for reproducing the experiments in ASTBERT: A Fine-tuned BERT model for Programming Languages based on ASTs. Unlike many other AST-based tools ASTBERT uses the subtrees of ASTs to extract syntactical information from programming languages.

For code classification, unzip and release the csv training data in folder 'data'. 
For code clone detection untar and release the Json file  in folder 'data'.

tar -xvf data_Java_code_pairs.tar

# Dependency
pip install keras
pip install transformers
pip install tensflow==2.2
pip install keras-bert

# Run code classification

run python train.py

# run code clone detection

run python train_code_clone_detection.py
