
import keras_nlp
import keras
import tensorflow.data as tf_data
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime

BATCH_SIZE = 16
LEARNING_RATE=1e-4
EPOCHS = 20
EMBED_DIM = 256
INTERMEDIATE_DIM = 2048
NUM_HEADS = 8
MAX_SEQUENCE_LENGTH = 128
EN_VOCAB_SIZE = 30000
CS_VOCAB_SIZE = 30000

train_ds = tf_data.Dataset.load("datasets/preprocessed_europarl_train")
valid_ds = tf_data.Dataset.load("datasets/preprocessed_europarl_valid")

# Encoder
encoder_inputs = keras.Input(shape=(None,), name="encoder_inputs")

x = keras_nlp.layers.TokenAndPositionEmbedding(
    vocabulary_size=EN_VOCAB_SIZE,
    sequence_length=MAX_SEQUENCE_LENGTH,
    embedding_dim=EMBED_DIM,
)(encoder_inputs)

encoder_outputs = keras_nlp.layers.TransformerEncoder(
    intermediate_dim=INTERMEDIATE_DIM, num_heads=NUM_HEADS
)(inputs=x)
encoder = keras.Model(encoder_inputs, encoder_outputs)


# Decoder
decoder_inputs = keras.Input(shape=(None,), name="decoder_inputs")
encoded_seq_inputs = keras.Input(shape=(None, EMBED_DIM), name="decoder_state_inputs")

x = keras_nlp.layers.TokenAndPositionEmbedding(
    vocabulary_size=CS_VOCAB_SIZE,
    sequence_length=MAX_SEQUENCE_LENGTH,
    embedding_dim=EMBED_DIM,
)(decoder_inputs)

x = keras_nlp.layers.TransformerDecoder(
    intermediate_dim=INTERMEDIATE_DIM, num_heads=NUM_HEADS
)(decoder_sequence=x, encoder_sequence=encoded_seq_inputs)
x = keras.layers.Dropout(0.5)(x)
decoder_outputs = keras.layers.Dense(CS_VOCAB_SIZE, activation="softmax")(x)
decoder = keras.Model(
    [
        decoder_inputs,
        encoded_seq_inputs,
    ],
    decoder_outputs,
)
decoder_outputs = decoder([decoder_inputs, encoder_outputs])

transformer = keras.Model(
    [encoder_inputs, decoder_inputs],
    decoder_outputs,
    name="transformer",
)

transformer.summary()

optimizer = Adam(learning_rate=LEARNING_RATE)
transformer.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
model_checkpoint = ModelCheckpoint(f'models_europarl/en_cs_translator_checkpoint_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.keras', save_best_only=True)

transformer.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=valid_ds,
    batch_size=BATCH_SIZE,
    callbacks=[early_stopping, model_checkpoint]
)
transformer.save(f'models_europarl/en_cs_translator_saved_{datetime.datetime.now().strftime("%Y%m%d_%H%M")}.keras')

