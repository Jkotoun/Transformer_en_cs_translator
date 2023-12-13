
import keras_nlp
import keras
import tensorflow.data as tf_data
import pickle
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
import random
import re
from sacrebleu.metrics import CHRF, BLEU
import time
import sys
# from keras import ops
#hyperparameters
MAX_SEQUENCE_LENGTH = 64
eval_samples = 10

transformer = keras.models.load_model('models_europarl/en_cs_translator_saved_20231209_0046.keras')
def read_files(path, lowercase = False):
    with open(path, "r", encoding="utf-8") as f:
        dataset_split = f.read().split("\n")[:-1]
    #to lowercase, idk why
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

def compute_probabilities(logits):
    return keras.activations.softmax(logits)

def next_token_logits(encoder_input_tokens, prompt, predicted_token_index):
    logits =  transformer(
        [tf.expand_dims(encoder_input_tokens, axis=0), tf.expand_dims(prompt, axis=0)]
    )[:, predicted_token_index-1, :] #we need prediction for next token, which is on index of last generated token
    return logits


def greedy_decode(encoder_input_tokens, prompt, end_token_id):
    
    start_index = 1
    current_prompt = prompt
    for predicted_token_index in range(start_index, MAX_SEQUENCE_LENGTH):
        next_logits = next_token_logits(encoder_input_tokens, current_prompt, predicted_token_index)
        next_probabilities = compute_probabilities(next_logits)
        max_probability_token_id = tf.argmax(next_probabilities, axis=-1) #index in logits array is equal to id
        indices = tf.constant([[predicted_token_index]])
        data = tf.constant([max_probability_token_id.numpy()[0]])
        current_prompt = tf.tensor_scatter_nd_update(current_prompt, indices, data)
        #generated end token
        if max_probability_token_id == end_token_id:
            break
    return current_prompt
    


def beam_decode(encoder_input_tokens, prompt, end_token_id, beam_size):
    start_index = 1
    #initial beam
    next_logits = next_token_logits(encoder_input_tokens, prompt, start_index)
    next_probabilities = compute_probabilities(next_logits)
    top_k_probabilities, top_k_token_indices = tf.math.top_k(next_probabilities, k=beam_size)
    current_subsequencies = []
    for index, value in enumerate(top_k_token_indices.numpy()[0]):
        #add to current subsequencies 5 versions of prompt with top k tokens on index 1
        indices = tf.constant([[start_index]])
        data = tf.constant([value])
        current_prompt = tf.tensor_scatter_nd_update(prompt, indices, data)
        #add potential subsequence with its log probability and length-normalized log probability (here length = 1, so its same)
        log_prob = tf.math.log(top_k_probabilities.numpy()[0][index])
        current_subsequencies.append((current_prompt, log_prob, log_prob))

    final_potential_solutions = []
    for predicted_token_index in range(start_index+1, MAX_SEQUENCE_LENGTH):
        #solutions which generated end token
        if len(final_potential_solutions) == beam_size:
            break

        tmp_subsequencies = []
        for index, (subseq_prompt, subseq_log_probability, _) in enumerate(current_subsequencies):
            next_logits = next_token_logits(encoder_input_tokens, subseq_prompt, predicted_token_index)
            next_probabilities = compute_probabilities(next_logits)
            top_k_probabilities, top_k_token_indices = tf.math.top_k(next_probabilities, k=beam_size-len(final_potential_solutions))
            for index, value in enumerate(top_k_token_indices.numpy()[0]):
                #add to current subsequencies 5 versions of prompt with top k tokens on index 1
                indices = tf.constant([[predicted_token_index]])
                data = tf.constant([value])
                updated_subseq_prompt = tf.tensor_scatter_nd_update(subseq_prompt, indices, data)
                #add potential subsequence with its log probability
                nextLogProbability = tf.math.log(top_k_probabilities.numpy()[0][index])
                tmp_subsequencies.append((updated_subseq_prompt, subseq_log_probability + nextLogProbability, (subseq_log_probability + nextLogProbability)/(predicted_token_index+1)))
        
        current_subsequencies = []
        current_sequences_to_find = beam_size - len(final_potential_solutions)
        tmp_subsequencies = sorted(tmp_subsequencies, key=lambda x: x[2], reverse=True)
        for i in range(current_sequences_to_find):
            if tmp_subsequencies[i][0][predicted_token_index] == end_token_id:
                final_potential_solutions.append(tmp_subsequencies[i])
            else:
                current_subsequencies.append(tmp_subsequencies[i])
    
    #get best 
    final_potential_solutions = sorted(final_potential_solutions, key=lambda x: x[2], reverse=True)

    if len(final_potential_solutions) > 0:
        return final_potential_solutions[0][0]
    #didnt generate any probable sequence to end
    else:
        sorted_subs = sorted(current_subsequencies, key=lambda x: x[2], reverse=True)
        return sorted_subs[0][0]


def decode_sequences(input_sentence):

    # Tokenize the encoder input.
    encoder_input_tokens = en_tokenizer(input_sentence)
    # encoder_input_tokens = tf.expand_dims(encoder_input_tokens, axis=0)
    if len(encoder_input_tokens) < MAX_SEQUENCE_LENGTH:
        pads = tf.fill((MAX_SEQUENCE_LENGTH - len(encoder_input_tokens)), 0)
        encoder_input_tokens = tf.concat([encoder_input_tokens, pads], 0)
    if len(encoder_input_tokens) > MAX_SEQUENCE_LENGTH:
        encoder_input_tokens = encoder_input_tokens[:MAX_SEQUENCE_LENGTH]
        print(f"Warning, input sentence was longer than max tokens limit", file=sys.stderr)

    start = tf.fill((1), cs_tokenizer.token_to_id("[START]"))
    pads = tf.fill((MAX_SEQUENCE_LENGTH - 1), cs_tokenizer.token_to_id("[PAD]"))
    prompt = tf.concat((start, pads), axis=-1)

    end_token_id = cs_tokenizer.token_to_id("[END]")

    generated_tokens = greedy_decode(encoder_input_tokens, prompt, end_token_id)
    # generated_tokens = beam_decode(encoder_input_tokens, prompt, end_token_id, 5)
    
    generated_sentences = cs_tokenizer.detokenize(tf.expand_dims(generated_tokens, axis=0))
    return generated_sentences


test_en = read_files('datasets/europarl/test-cs-en.en')
test_cs = read_files('datasets/europarl/test-cs-en.cs')

chrf = CHRF() 
bleu = BLEU()
refs = test_cs[:eval_samples]
translations = []
start_time = time.time()

for i in range(len(refs)):

    cs_translated = decode_sequences(test_en[i])
    cs_translated = cs_translated.numpy()[0].decode("utf-8")
    cs_translated = (
        cs_translated.replace("[PAD]", "")
        .replace("[START]", "")
        .replace("[END]", "")
        .strip()
    )
    #remove spaces before interpunction
    cs_translated = re.sub(r'\s+([.,;!?:])', r'\1', cs_translated)
    print(cs_translated, flush=True)
    translations.append(cs_translated)

end_time = time.time()



print("evaluating bleu", flush=True)

refs_twodim = [[ref] for ref in refs]

print("evaluating chrf", flush=True)
chrf2_result = chrf.corpus_score(translations, refs_twodim)
bleu_result = bleu.corpus_score(translations, refs_twodim)
print("chrf2")
print(chrf2_result)
print("bleu")
print(bleu_result)
elapsed_time = end_time - start_time
print("elapsed time")
print(elapsed_time)