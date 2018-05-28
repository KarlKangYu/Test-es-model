import re


class DataUtility:
    def __init__(self, vocab_file_in_words=None, vocab_file_in_letters=None, vocab_file_out=None, corpus_file=None):
        
        self.unk_str = "<unk>"
        self.num_str = "<num>"
        self.pun_str = "<pun>"
        self.start_str = "<start>"
        self.pad_id = 0

        if vocab_file_in_words and vocab_file_in_letters and vocab_file_out:
            self.id2token_in, self.id2token_out = {}, {}
            self.token2id_in_words, self.token2id_in_letters, self.token2id_out = {}, {}, {}
            with open(vocab_file_in_words, mode="r") as f:
                for line in f:
                    token, id = line.strip().split("##")
                    id = int(id)
                    self.id2token_in[id] = token
                    self.token2id_in_words[token] = id
            print ("in words vocabulary size =", str(len(self.token2id_in_words)))
            with open(vocab_file_in_letters, mode="r") as f:
                for line in f:
                    token, id = line.strip().split("##")
                    id = int(id)
                    self.id2token_in[id] = token
                    self.token2id_in_letters[token] = id
            print ("in letters vocabulary size =", str(len(self.token2id_in_letters)))
            print ("in vocabulary size =", str(len(self.id2token_in)))
            with open(vocab_file_out, mode="r") as f:
                for line in f:
                    token, id = line.split("##")
                    id = int(id)
                    self.id2token_out[id] = token
                    self.token2id_out[token] = id
            print ("out vocabulary size =", str(len(self.token2id_out)))

            self.start_id = self.token2id_in_letters[self.start_str]

    def word2id(self, word):
        if re.match("^[a-zA-Z]$", word) or (word in self.token2id_in_words):
            word_out = word
        else:
            if re.match("^[+-]*[0-9]+.*[0-9]*$", word):
                word_out = self.num_str
            else:
                if re.match("^[^a-zA-Z0-9']*$", word):
                    word_out = self.pun_str
                else:
                    word_out = self.unk_str
        return self.token2id_in_words.get(word_out, self.token2id_in_words[self.unk_str])

    def words2ids(self, words):
        words_split = re.split("\\s+", words)
        return [self.word2id(word) for word in words_split if len(word) > 0]

    def letters2ids(self, letters):
        letters_split = re.split("\\s+", letters)
        return [self.start_id] + [self.token2id_in_letters.get(letter, self.token2id_in_letters[self.unk_str])
                                  for letter in letters_split if len(letter) > 0]

    def outword2id(self, outword):
        return self.token2id_out.get(outword, self.token2id_out[self.unk_str])

    def ids2outwords(self, ids_out):
        return [self.id2token_out.get(id, self.unk_str) for id in ids_out]

    def ids2inwords(self, ids_in):
        return [self.id2token_in.get(int(id), self.unk_str) for id in ids_in]

    def data2ids_line(self, data_line):
        data_line_split = re.split("\\t+", data_line)#分成三部分，第一部分是上下文，第二部分是要预测的部分的键码部分，第三部分是要预测的真实单词
        words = data_line_split[0].strip()
        letters = data_line_split[1].strip()
        letters = letters[3 : len(letters) - 4].strip()#去除掉前后的<b>，变成单纯的键码部分
        words_ids = self.words2ids(words)
        letters_ids = self.letters2ids(letters)
        words_num = len(words_ids)
        letters_num = len(letters_ids)
        return words_ids, letters_ids, words_num, letters_num

    def sentence2ids(self, sentence):
        words_array = re.split('\\s+', sentence)
        word_letters = words_array[-1]
        words_array = words_array[:-1]
        words = ' '.join(words_array)
        letters = ' '.join(word_letters)
        words_ids = self.words2ids(words)
        letters_ids = self.letters2ids(letters)
        return words_ids, letters_ids, word_letters

