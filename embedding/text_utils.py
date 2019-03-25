import re
import random
import numpy as np
import torch
import torch.utils.data
import multiprocessing
import os
import torchvision.transforms as transforms

def clean_report(reports, clean=1):
    
    clean_reports = []
    vocab_report = set()
    
    replace = {"\n": " ", "~sbld~": " ", "~ebld~": " "}
    replace = dict((re.escape(k), v) for k, v in replace.items())

    if clean == 1:
        #clean newline characters from report and return cleaned report
        for report in reports:
            if report is not None:
                pattern = re.compile("|".join(replace.keys()))
                report = pattern.sub(lambda m: replace[re.escape(m.group(0))], report)
                clean_reports.append(report)
            else:
                print("A report is None")
    elif clean == 2:
        #clean words in report and return report as a list of words
        try:
            reports = [report.split(' ') for report in reports]
        except Exception as e:
            print("A report is None")
        regex = re.compile('[^a-zA-Z0-9]')
        for report in reports:
            if report is not None:
                clean_words = []
                for word in report:
                    if word is not '':
                        _word = regex.sub('', word)
                        if _word is not '':
                            clean_words.append(_word)
                            vocab_report.add(_word)
                clean_reports.append(clean_words)
            else:
                print("A report is None")
    return clean_reports, vocab_report


def shuffle_lists(list1, list2):
    
    temp_list = list(zip(list1, list2))
    random.shuffle(temp_list)
    
    list1, list2 = zip(*temp_list)
    
    return list1, list2


def get_max_lengths(reports, filename):
    
    vocab = load_glove_vocab(filename)
    tmp = []
    for report in reports:
        tmp.append([word for word in report if word in vocab])
    
    max_length_word = max([max(map(lambda x: len(x), report)) for report in tmp])
    max_length_report= max(map(lambda x : len(x), tmp))
    
    return max_length_report, max_length_word


def load_glove_vocab(filename):
    
    file = open(filename, 'r')
    
    #create GloVe vocab first
    vocab_glove = set()
    for line in file.readlines():
        line = line.strip().split(' ')
        word = line[0]
        vocab_glove.add(word)
    
    return vocab_glove


def convert_report(reports, processing_word):
    
    clean_reports, _ = clean_report(reports)
    clean_reports, _ = clean_report(clean_reports, 2)
    
    report_id_form = []
    for report in clean_reports:
        words = []
        for word in report:
            word_tuple = processing_word(word)
            if word_tuple:
                words.append(word_tuple)
        report_id_form.append(words)
    return report_id_form


def get_char_vocab(dataset):
    
    #build char vocab
    vocab_chars =set()
    for report in dataset:
        for word in report:
            vocab_chars.update(word)

    #char vocab to dict
    d = dict()
    for idx, char in enumerate(vocab_chars):
        d[char] = idx
    vocab_chars = d
    
    return vocab_chars


#function has been checked
def load_glove(filename, vocab_report, dim):
    """
    Return vocabularly and GloVe embedding vectors from given file name.
    Args:
        filename containing GloVe vectors
        vocabulary of words from report
        dimension of GloVe Vectors
    Returns:
        embeddings: dictionary, such that embeddings[index] = GloVe vector
    """
    
    file = open(filename, 'r')
    
    #create GloVe vocab first
    vocab_glove = set()
    for line in file.readlines():
        line = line.strip().split(' ')
        word = line[0]
        vocab_glove.add(word)
    
    #discards words not in glove (future: include these words)
    vocab = vocab_glove & vocab_report
    
    d = dict()
    for idx, word in enumerate(vocab):
        word = word.strip()
        d[word] = idx
    vocab = d
        
    embeddings = np.zeros([len(vocab), dim])
    file = open(filename, 'r')
    for line in file.readlines():
        line = line.strip().split(' ')
        word = line[0]
        embedding = [float(x) for x in line[1:]]
        if word in vocab:
            word_id = vocab[word]
            embeddings[word_id] = np.asarray(embedding)    
    print ('loaded GloVe')
    file.close()
    
    return vocab, embeddings


def save_to_file(vocab, filename):
    
    print('Writing vocab...')
    with open(filename, "w") as f:
        for i, word in enumerate(vocab):
            if i != len(vocab) - 1:
                f.write("{}\n".format(word))
            else:
                f.write(word)
    print("Done. {} tokens".format(len(vocab)))
    
    
def load_vocab(filename):
    
    print('Opening vocab...')
    d = dict()
    with open(filename, 'r') as f:
        for idx, word in enumerate(f):
            word = word.strip()
            d[word] = idx
    return d


def load_dict(filename):
    
    print('Opening vocab...')
    d = dict()
    with open(filename, 'r') as f:
        for idx, word in enumerate(f):
            word = word.strip()
            d[idx] = word
    return d


def load_npz(filename, name):
    
    with np.load(filename) as data:
        return data[name]
    
    
def get_processing_word(vocab_words=None, vocab_chars=None,
                    lowercase=False, chars=False):
    """Return lambda function that transform a word (string) into list,
    or tuple of (list, id) of int corresponding to the ids of the word and
    its corresponding characters.
    Args:
        vocab: dict[word] = idx
    Returns:
        f("cat") = ([12, 4, 32], 12345)
                 = (list of char ids, word id)
    """
    def f(word):
        # 0. get chars of words
        if vocab_chars is not None and chars == True:
            char_ids = []
            for char in word:
                # ignore chars out of vocabulary
                if char in vocab_chars:
                    char_ids += [vocab_chars[char]]

        # 1. preprocess word
        if lowercase:
            word = word.lower()

        # 2. get id of word
        if vocab_words is not None:
            if word in vocab_words:
                word = vocab_words[word]
                # 3. return tuple char ids, word id
                if vocab_chars is not None and chars == True:
                    return char_ids, word
                else:
                    return word
            else:
                #print("Word discarded: %s"%word)
                return None

    return f


def get_minibatches(data, minibatch_size):
    result = []
    for item in data:
        result.append(item)
        if len(result) == minibatch_size:
            yield tuple(result)
            result = []
    if len(result) > 0:
        yield tuple(result)
        
        
def convert_report(reports, processing_word):
    
    clean_reports, _ = clean_report(reports)
    clean_reports, _ = clean_report(clean_reports, 2)
    
    report_id_form = []
    for report in clean_reports:
        words = []
        for word in report:
            word_tuple = processing_word(word)
            if word_tuple:
                words.append(word_tuple)
        report_id_form.append(words)
    return report_id_form


def split_report(reports):
    _word_ids = []
    _char_ids = []
    for report in reports:
        _words = []
        _char = []
        for word in report:
            _words.append(word[1])
            _char.append(word[0])
        _word_ids.append(_words)
        _char_ids.append(_char)   
    return _word_ids, _char_ids


def pad_reports(reports, pad_tok, max_length_word=None, max_length_report=None, is_char_ids=False):
    
    if is_char_ids:
        if max_length_word is None:
            max_length_word = max([max(map(lambda x: len(x), report)) for report in reports])
        report_padded, report_length = [], []
        for report in reports:
            rp, rl = _pad_reports(report, pad_tok, max_length_word)
            report_padded.append(rp)
            report_length.append(rl)
        if max_length_report is None:
            max_length_report = max(map(lambda x: len(x), reports))
        report_padded, _ = _pad_reports(report_padded, 
                                [pad_tok]*max_length_word, max_length_report)
        report_length, _ = _pad_reports(report_length, 0, max_length_report)
        
    else:
        if max_length_report is None:
            max_length_report= max(map(lambda x : len(x), reports))
        report_padded, report_length = _pad_reports(reports,
                                            pad_tok, max_length_report)

    return report_padded, report_length

def _pad_reports(reports, pad_tok, max_length):

    report_padded, report_length = [], []

    for report in reports:
        report_ = report[:max_length] + [pad_tok]*max(max_length - len(report), 0)
        report_padded +=  [report_]
        report_length += [min(len(report), max_length)]

    return report_padded, report_length

def create_report_examples():
    raw_reports = np.load('/home/rohanmirchandani/maxwell-pt-test/points.npy')
    dirty_reports = [report['body'] for report in raw_reports]
    clean_reports, _ = tu.clean_report(dirty_reports, clean=1) # first pass removes \n's and weird characters
    tokenised_reports, report_vocab = tu.clean_report(clean_reports, clean=2) # second pass tokenises and builds vocab
    vocab, embeddings = tu.load_glove('/home/rohanmirchandani/glove/glove.6B.50d.w2vformat.txt', report_vocab, 50)
    for i, tokens in enumerate(tokenised_reports): # should multithread this at some point
        print(i)
        vecs = np.array([embeddings[vocab[token]] for token in tokens if token in vocab.keys()])
        data = data = {'tokens': tokens, "vectors": vecs}
        name = "example_{}".format(i)
        np.save(os.path.join('/home/rohanmirchandani/maxwell-pt-test/examples/', name), data)
        
class ReportDataset(torch.utils.data.Dataset):
    
    def __init__(self, root):
        self.root = root

    def __len__(self):
        return len(os.listdir(self.root))
        
    def __getitem__(self, idx):
        fns = os.listdir(self.root)
        fn = os.path.join(self.root, fns[idx])
        sample = np.load(fn).item()
        tokens = sample['tokens']
        if not sample['vectors'].shape:
            sample['vectors'] = sample['vectors'].reshape(-1, 50)
        vecs = sample['vectors']
        return tokens, vecs
        
def create_dataloader(directory, batch_size=1, shuffle=True, workers=8):
    dataset = ReportDataset(root=directory)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers)
    return dataloader

def PCA(data, k=2):
    X = torch.from_numpy(data)
    X_mean = torch.mean(X, 0)
    X = X - X_mean.expand_as(X)
    U, S, V = torch.svd(X.transpose(0, 1))
    return torch.mm(X, U[:, :k])
