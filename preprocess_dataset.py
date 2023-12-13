
import keras_nlp
import tensorflow.data as tf_data
#hyperparameters
BATCH_SIZE = 16
MAX_SEQUENCE_LENGTH = 64

def read_files(path, lowercase = False):
    with open(path, "r", encoding="utf-8") as f:
        dataset_split = f.read().split("\n")[:-1]
    if(lowercase):
        dataset_split = [line.lower() for line in dataset_split]
    return dataset_split

en_vocab = read_files("tokenizers/en_europarl_vocab")
cs_vocab = read_files("tokenizers/cs_europarl_vocab")

en_tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(
    vocabulary=en_vocab, 
    lowercase=False
)
cs_tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(
    vocabulary=cs_vocab, 
    lowercase=False
)


#europarl
train_cs_file = 'datasets/europarl/train-cs-en.cs'
train_en_file = 'datasets/europarl/train-cs-en.en'
valid_cs_file = 'datasets/europarl/valid-cs-en.cs'
valid_en_file = 'datasets/europarl/valid-cs-en.en'


train_cs = read_files(train_cs_file, True)
train_en = read_files(train_en_file, True)
valid_cs = read_files(valid_cs_file, True)
valid_en = read_files(valid_en_file, True)

def preprocess_batch(en, cs):
    en = en_tokenizer(en)
    cs = cs_tokenizer(cs)

    en_start_end_packer = keras_nlp.layers.StartEndPacker(
        sequence_length=MAX_SEQUENCE_LENGTH,
        pad_value=en_tokenizer.token_to_id("[PAD]"),
    )
    en = en_start_end_packer(en)

    cs_start_end_packer = keras_nlp.layers.StartEndPacker(
        sequence_length=MAX_SEQUENCE_LENGTH + 1,
        start_value=cs_tokenizer.token_to_id("[START]"),
        end_value=cs_tokenizer.token_to_id("[END]"),
        pad_value=cs_tokenizer.token_to_id("[PAD]"),
    )
    cs = cs_start_end_packer(cs)

    return (
        {
            "encoder_inputs": en,
            "decoder_inputs": cs[:, :-1],
        },
        cs[:, 1:],
    )


def make_dataset(en_texts, cs_texts):
    dataset = tf_data.Dataset.from_tensor_slices((en_texts, cs_texts))
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.map(preprocess_batch, num_parallel_calls=tf_data.AUTOTUNE)
    return dataset.shuffle(2048).prefetch(16).cache()


train_ds = make_dataset(train_en, train_cs)
val_ds = make_dataset(valid_en, valid_cs)

tf_data.Dataset.save(train_ds, "datasets/preprocessed_europarl_train")
tf_data.Dataset.save(val_ds, "datasets/preprocessed_europarl_valid")




