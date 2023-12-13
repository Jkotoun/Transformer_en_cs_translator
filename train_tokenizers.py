import keras_nlp
import tensorflow.data as tf_data

EN_VOCAB_SIZE = 30000
CS_VOCAB_SIZE = 30000

def train_word_piece(text_samples, vocab_size, reserved_tokens, save_output_path):
    word_piece_ds = tf_data.Dataset.from_tensor_slices(text_samples)
    vocab = keras_nlp.tokenizers.compute_word_piece_vocabulary(
        word_piece_ds.batch(1000).prefetch(2),
        vocabulary_size=vocab_size,
        reserved_tokens=reserved_tokens,
        vocabulary_output_file=save_output_path
    )
    return vocab

def read_files(path):
    with open(path, "r", encoding="utf-8") as f:
        dataset_split = f.read().split("\n")[:-1]
    dataset_split = [line.lower() for line in dataset_split]
    return dataset_split

train_cs = read_files('datasets/europarl/train-cs-en.cs')
train_en = read_files('datasets/europarl/train-cs-en.en')
reserved_tokens = ["[PAD]", "[UNK]", "[START]", "[END]"]
en_vocab = train_word_piece(train_en, EN_VOCAB_SIZE, reserved_tokens, "tokenizers/en_europarl_vocab")
cs_vocab = train_word_piece(train_cs, CS_VOCAB_SIZE, reserved_tokens, "tokenizers/cs_europarl_vocab")



