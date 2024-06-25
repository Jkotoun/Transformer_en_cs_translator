# English to czech translator
Simple english to czech seq2seq translator based on transformer architecture. The transformer model is built using Keras_NLP transformer encoder, transformer decoder and positional encoder. Decoding algorithms were implemented from scratch (Greedy and Beam search). Repository contains scripts to preprocess the europarl dataset, train WordPiece tokenizer, train the model and evaluate it. 

## To train model and run evaluation:

Download and extract the europarl cs_en from https://www.statmt.org/europarl/ to datasets/europarl/ folder

Install requirements using pip install -r requirements.txt

Run scripts in this order:
- create_dataset_splits.py
- train_tokenizers.py
- preprocess_dataset.py
- train.py
- eval.py
