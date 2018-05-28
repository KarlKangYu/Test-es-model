#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import sys
import os
import numpy as np
import time
from data_utility_old import DataUtility
from config import Config
import tensorflow as tf


class InputEngineRnn:

    def __init__(self, graph_path, vocab_path, config_name):

        vocab_file_in_words = os.path.join(vocab_path, "vocab_in_words")
        vocab_file_in_letters = os.path.join(vocab_path, "vocab_in_letters")
        vocab_file_out = os.path.join(vocab_path, "vocab_out")

        self._config = Config()
        self._config.get_config(config_name)
        self._data_utility = DataUtility(vocab_file_in_words=vocab_file_in_words, vocab_file_in_letters=vocab_file_in_letters,
                                         vocab_file_out=vocab_file_out)
        self.sparsity = self._config.sparsity
        prefix = "import/"
        self.lm_top_k_name = prefix + "Online/Model/top_k:0"
        self.lm_state_in_name = prefix + "Online/Model/state:0"
        self.lm_input_name = prefix + "Online/Model/batched_input_word_ids:0"

        self.lm_top_k_prediction_name = prefix + "Online/Model/top_k_prediction:1"
        self.lm_output_name = prefix + "Online/Model/probabilities:0"
        self.lm_state_out_name = prefix + "Online/Model/state_out:0"

        self.lm_output_state = prefix + "Online/KeyModel/lm_out_state:0"
        self.kc_top_k_name = prefix + "Online/KeyModel/top_k:0"
        self.kc_state_in_name = prefix + "Online/KeyModel/state:0"
        self.kc_input_name = prefix + "Online/KeyModel/batched_input_word_ids:0"

        self.kc_top_k_prediction_name = prefix + "Online/KeyModel/top_k_prediction:1"
        self.kc_output_name = prefix + "Online/KeyModel/probabilities:0"
        self.kc_state_out_name = prefix + "Online/KeyModel/state_out:0"

        

        with open(graph_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def)
        gpu_config = tf.ConfigProto()
        gpu_config.gpu_options.per_process_gpu_memory_fraction = self._config.gpu_fraction
        self._sess = tf.Session(config=gpu_config)

    def predict(self, sentence, k):
        global probabilities, top_k_predictions
        inputs, inputs_key, word_letters = self._data_utility.sentence2ids(sentence)
        print (inputs, inputs_key, word_letters)
        lm_state = np.zeros([self._config.num_layers, 2, 1, self._config.lm_hidden_size], dtype=np.float32)
        kc_state = np.zeros([self._config.num_layers, 2, 1, self._config.kc_hidden_size], dtype=np.float32)
        if len(inputs) > 0:
            for i in range(len(inputs)):
                feed_values = {self.lm_input_name: [[inputs[i]]]}
                if i > 0:
                    feed_values[self.lm_state_in_name] = lm_state
                # probabilities is an ndarray of shape (batch_size * time_step) * vocab_size
                # For inference, batch_size = num_step = 1, thus probabilities.shape = 1 * vocab_size
                result = self._sess.run([self.lm_state_out_name], feed_dict=feed_values)
                lm_state = result[0]
                #probability_topk = [probabilities[0][id] for id in top_k_predictions[0]]
                #words_out = self._data_utility.ids2outwords(top_k_predictions[0])

        for i in range(len(inputs_key)):
            feed_values = {self.kc_input_name: [[inputs_key[i]]],
                           self.kc_top_k_name: k}
            if i > 0 or len(inputs) == 0:
                feed_values[self.kc_state_in_name] = kc_state
            else:
                feed_values[self.lm_output_state] = lm_state
            
            probabilities, top_k_predictions, kc_state = self._sess.run([self.kc_output_name, self.kc_top_k_prediction_name,
                                                                          self.kc_state_out_name], feed_dict=feed_values)
            
            probability_topk = [probabilities[0][id] for id in top_k_predictions[0]]
            words_out = self._data_utility.ids2outwords(top_k_predictions[0])

        return [{'word': word, 'probability': float(probability)}
                if word != '<unk>' else {'word': '<' + word_letters + '>', 'probability': float(probability)}
                for word, probability in zip(words_out, probability_topk)] if len(words_out) > 0 else []

    def predict_data(self, sentence, k):
        sentence = sentence.rstrip()
        inputs, inputs_key, words_num, letters_num = self._data_utility.data2ids_line(sentence)#上下文的id，要预测的单词的键码部分id，上下文单词数，要预测的单词的字母数
        words_out = []
        lm_state = np.zeros([self._config.num_layers, 2, 1, self._config.word_hidden_size], dtype=np.float32)
        kc_state = np.zeros([self._config.num_layers, 2, 1, self._config.letter_hidden_size], dtype=np.float32)
        if len(inputs) > 0:
            for i in range(len(inputs)):
                feed_values = {self.lm_input_name: [[inputs[i]]]}
                if i > 0:
                    feed_values[self.lm_state_in_name] = lm_state
                # probabilities is an ndarray of shape (batch_size * time_step) * vocab_size
                # For inference, batch_size = num_step = 1, thus probabilities.shape = 1 * vocab_size
                result = self._sess.run([self.lm_state_out_name], feed_dict=feed_values)
                lm_state = result[0]
                #probability_topk = [probabilities[0][id] for id in top_k_predictions[0]]
                #words = self._data_utility.ids2outwords(top_k_predictions[0])
                #words_out.append(words)

        for i in range(len(inputs_key)):
            feed_values = {self.kc_input_name: [[inputs_key[i]]],
                           self.kc_top_k_name: k}
            if i > 0 or len(inputs) == 0:
                feed_values[self.kc_state_in_name] = kc_state
            else:
                feed_values[self.lm_output_state] = lm_state
            #print (state_out)
            probabilities, top_k_predictions, kc_state = self._sess.run([self.kc_output_name, self.kc_top_k_prediction_name,
                                                                          self.kc_state_out_name], feed_dict=feed_values)
            probability_topk = [probabilities[0][id] for id in top_k_predictions[0]]
            words = self._data_utility.ids2outwords(top_k_predictions[0])
            words_out.append(words)
        out_str = str(words_out if words_num > 0 else [['', '', '']] + words_out[1: ])
        return out_str

    def predict_file(self, test_file_in, test_file_out, k):
        testfilein = open(test_file_in, "r")
        testfileout = open(test_file_out, 'w')
        t1 = time.time()
        topk = k
        for sentence in testfilein:
            sentence = sentence.rstrip()
            out_str = self.predict_data(sentence, topk)
            if (out_str):
                print (sentence + " | " + out_str)
                testfileout.write(sentence + " | " + out_str + "\n")
            else:
                print ("predict error : " + sentence)
        t2 = time.time()
        print(t2 - t1)
        testfilein.close()
        testfileout.close()

if __name__ == "__main__":
    args = sys.argv

    graph_path = args[1]
    vocab_path = args[2]
    config_name = args[3]
    engine = InputEngineRnn(graph_path, vocab_path, config_name)
    test_file_in = args[4]
    test_file_out = "test_result"
    engine.predict_file(test_file_in, test_file_out, 3)


    #while True:
       # sentence = raw_input("please enter sentence:")
        #if sentence == "quit()":
       #     exit()
        #res = engine.predict(sentence, 10)
       # print(sentence)
       # print(str(res).decode('string_escape'))

