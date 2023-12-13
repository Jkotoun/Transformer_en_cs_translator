import keras_nlp
import keras
import tensorflow.data as tf_data
import pickle
import random

def read_files(path):
    with open(path, "r", encoding="utf-8") as f:
        dataset_split = f.read().split("\n")[:-1]
    dataset_split = [line.lower() for line in dataset_split]
    return dataset_split

def save_list_to_file(file_path, string_list):
    with open(file_path, 'w') as file:
        file.writelines([f"{string}\n" for string in string_list])

#load files
cs_file = 'datasets/europarl/europarl-v7.cs-en.cs'
en_file = 'datasets/europarl/europarl-v7.cs-en.en'
sentences_cs = read_files(cs_file)
sentences_en = read_files(en_file)

#create pairs and split to train, valid and test
pairs = list(zip(sentences_en, sentences_cs))
random.shuffle(pairs)
num_val_samples = int(0.15 * len(pairs))
num_train_samples = len(pairs) - 2 * num_val_samples

train_pairs = pairs[:num_train_samples]
valid_pairs = pairs[num_train_samples : num_train_samples + num_val_samples]
test_pairs = pairs[num_train_samples + num_val_samples :]

print(train_pairs[0])
print(valid_pairs[0])
print(test_pairs[0])


en_train_samples = [pair[0] for pair in train_pairs]
cs_train_samples = [pair[1] for pair in train_pairs]
en_valid_samples = [pair[0] for pair in valid_pairs]
cs_valid_samples = [pair[1] for pair in valid_pairs]
en_test_samples = [pair[0] for pair in test_pairs]
cs_test_samples = [pair[1] for pair in test_pairs]


save_list_to_file("datasets/europarl/train-cs-en.en", en_train_samples)
save_list_to_file("datasets/europarl/train-cs-en.cs", cs_train_samples)
save_list_to_file("datasets/europarl/valid-cs-en.en", en_valid_samples)
save_list_to_file("datasets/europarl/valid-cs-en.cs", cs_valid_samples)
save_list_to_file("datasets/europarl/test-cs-en.en", en_test_samples)
save_list_to_file("datasets/europarl/test-cs-en.cs", cs_test_samples)


