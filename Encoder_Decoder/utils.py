# -*- coding: utf-8 -*-
import codecs
import nltk
import numpy as np
import cPickle as pickle

def batch_op(inputs, idx_pad, max_sequence_length=None):
    sequence_lengths = [len(seq) for seq in inputs]
    batch_size = len(inputs)

    if max_sequence_length is None:
        max_sequence_length = max(sequence_lengths)

    inputs_batch_major = np.ones([batch_size, max_sequence_length], np.int32) * idx_pad

    for i, seq in enumerate(inputs):
        for j, element in enumerate(seq):
            inputs_batch_major[i,j] = element

    inputs_time_major = inputs_batch_major.swapaxes(0,1)

    return inputs_time_major, sequence_lengths

def load_data_en(filename, vocab_to_idx, encoding=None):
    data = []
    with codecs.open(filename, 'r', encoding) as fin:
        for line in fin:
            data.append([vocab_to_idx.get(w, '<UNK>') for w in nltk.word_tokenize(line)])

    return data

def gen_vocab_dict(filename, encoding=None):
    vocabs = set(["<START>","<PAD>","<EOS>","<UNK>"])
    with codecs.open(filename, 'r', encoding) as fin:
        for line in fin:
            for w in nltk.word_tokenize(line):
                vocabs.add(w)

    vocab_to_idx = {v:i for i,v in enumerate(vocabs)}
    idx_to_vocab = {i:v for i,v in enumerate(vocabs)}

    return vocab_to_idx, idx_to_vocab

def save_vocab_dict_en(dict_path, vocab_to_idx, idx_to_vocab):
    if ".voc" not in dict_path:
        dict_path += ".voc"

    pickle.dump((vocab_to_idx, idx_to_vocab), open(dict_path, 'w'), protocol=pickle.HIGHEST_PROTOCOL)

def load_vocab_dict_en(dict_path):
    if ".voc" not in dict_path:
        dict_path += ".voc"

    vocab_to_idx, idx_to_vocab = pickle.load(open(dict_path,'r'))
    return vocab_to_idx, idx_to_vocab