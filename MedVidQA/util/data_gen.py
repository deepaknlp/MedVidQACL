import os
import codecs
import sys

import numpy as np
from tqdm import tqdm
import warnings
from collections import Counter
from nltk.tokenize import word_tokenize
from util.data_util import load_json, load_pickle, save_pickle, time_to_index

PAD, UNK = "<PAD>", "<UNK>"


class MedVidQAProcessor:
    def __init__(self):
        super(MedVidQAProcessor, self).__init__()
        self.idx_counter = 0

    def reset_idx_counter(self):
        self.idx_counter = 0

    def process_data(self, data, scope):
        results = []
        for index, data_item in tqdm(enumerate(data), total=len(data), desc='process medvidqa {}'.format(scope)):
            if len(data_item) == 0:
                warnings.warn("Warning: Skipping the dataset item as len(data_item)==0")
                continue
            video_id = data_item['video_id']
            start_time = data_item['answer_start_second']
            end_time = data_item['answer_end_second']
            duration = data_item['video_length']
            question = data_item['question']

            start_time = max(0.0, float(start_time))
            end_time = min(float(end_time), duration)
            words = word_tokenize(question.strip().lower(), language="english")
            record = {'sample_id': self.idx_counter, 'vid': str(video_id), 's_time': start_time, 'e_time': end_time,
                      'duration': duration, 'words': words}
            results.append(record)
            self.idx_counter += 1
        return results

    def convert(self, data_dir, test_json_data_file_name=None, eval=False):
        self.reset_idx_counter()
        if not os.path.exists(data_dir):
            raise ValueError('data dir {} does not exist'.format(data_dir))

        if eval and test_json_data_file_name is not None:
            test_data = load_json(os.path.join(data_dir, test_json_data_file_name))
            test_set = self.process_data(test_data, scope='test')
            return test_set

        #### for new data setting
        train_data = load_json(os.path.join(data_dir, 'train.json'))

        val_data = load_json(os.path.join(data_dir, 'val.json'))

        # process data
        train_set = self.process_data(train_data, scope='train')
        val_set = self.process_data(val_data, scope='val')
        test_set = None

        return train_set, val_set, test_set  # train/val/test


def load_glove(glove_path):
    vocab = list()
    with codecs.open(glove_path, mode="r", encoding="utf-8") as f:
        for line in tqdm(f, total=2196017, desc="load glove vocabulary"):
            line = line.lstrip().rstrip().split(" ")
            if len(line) == 2 or len(line) != 301:
                continue
            word = line[0].lower()
            vocab.append(word)
    return set(vocab)


def filter_glove_embedding(word_dict, glove_path):
    vectors = np.zeros(shape=[len(word_dict), 300], dtype=np.float32)
    with codecs.open(glove_path, mode="r", encoding="utf-8") as f:
        for line in tqdm(f, total=2196017, desc="load glove embeddings"):
            line = line.lstrip().rstrip().split(" ")
            if len(line) == 2 or len(line) != 301:
                continue
            word = line[0].lower()
            if word in word_dict:
                vector = [float(x) for x in line[1:]]
                word_index = word_dict[word]
                vectors[word_index] = np.asarray(vector)
    return np.asarray(vectors)


def vocab_emb_gen(datasets, emb_path):
    # generate word dict and vectors
    emb_vocab = load_glove(emb_path)
    word_counter, char_counter = Counter(), Counter()
    for data in datasets:
        for record in data:
            for word in record['words']:
                word_counter[word] += 1
                for char in list(word):
                    char_counter[char] += 1
    word_vocab = list()
    for word, _ in word_counter.most_common():
        if word in emb_vocab:
            word_vocab.append(word)
    tmp_word_dict = dict([(word, index) for index, word in enumerate(word_vocab)])
    vectors = filter_glove_embedding(tmp_word_dict, emb_path)
    word_vocab = [PAD, UNK] + word_vocab
    word_dict = dict([(word, idx) for idx, word in enumerate(word_vocab)])
    # generate character dict
    char_vocab = [PAD, UNK] + [char for char, count in char_counter.most_common() if count >= 5]
    char_dict = dict([(char, idx) for idx, char in enumerate(char_vocab)])
    return word_dict, char_dict, vectors


def dataset_gen(data, vfeat_lens, word_dict, char_dict, max_pos_len, scope):
    dataset = list()
    vid_not_found = 0
    word_not_found = 0
    total_word_count = 0

    for record in tqdm(data, total=len(data), desc='process {} data'.format(scope)):
        vid = record['vid']
        if vid not in vfeat_lens:
            vid_not_found += 1
            continue
        s_ind, e_ind, _ = time_to_index(record['s_time'], record['e_time'], vfeat_lens[vid], record['duration'])
        word_ids, char_ids = [], []
        for word in record['words'][0:max_pos_len]:
            if word in word_dict:
                word_id = word_dict[word]
                total_word_count += 1
            else:
                word_id = word_dict[UNK]
                word_not_found += 1
                total_word_count += 1

            char_id = [char_dict[char] if char in char_dict else char_dict[UNK] for char in word]
            word_ids.append(word_id)
            char_ids.append(char_id)

        result = {'sample_id': record['sample_id'], 'vid': record['vid'], 's_time': record['s_time'],
                  'e_time': record['e_time'], 'duration': record['duration'], 'words': record['words'],
                  's_ind': int(s_ind), 'e_ind': int(e_ind), 'v_len': vfeat_lens[vid], 'w_ids': word_ids,
                  'c_ids': char_ids}
        dataset.append(result)
    print(f"video ids not found: {vid_not_found}")
    print(f"Length of samples: {len(dataset)}")
    print(f"Word not found in pre-trained word embeddings {word_not_found}")
    print(f"Total words in data {total_word_count}")

    return dataset


def gen_or_load_dataset(configs):
    if not os.path.exists(configs.save_dir):
        os.makedirs(configs.save_dir)
    data_dir = os.path.join('data', 'dataset', configs.task)
    feature_dir = os.path.join('data', 'features', configs.task)
    if configs.suffix is None and configs.task == 'medvidqa':
        save_path = os.path.join(configs.save_dir, '_'.join([configs.task, str(configs.max_pos_len)]) +
                                 '.pkl')
    else:
        raise ValueError('Unknown task {}!!!'.format(configs.task))
    if os.path.exists(save_path):
        dataset = load_pickle(save_path)
        return dataset
    feat_len_path = os.path.join(feature_dir, 'feature_shapes.json')
    emb_path = os.path.join('data', 'word_embedding', 'glove.840B.300d.txt')
    # load video feature length
    vfeat_lens = load_json(feat_len_path)
    for vid, vfeat_len in vfeat_lens.items():
        vfeat_lens[vid] = min(configs.max_pos_len, vfeat_len)
    # load data

    if configs.task == 'medvidqa':
        processor = MedVidQAProcessor()
    else:
        raise ValueError('Unknown task {}!!!'.format(configs.task))
    train_data, val_data, test_data = processor.convert(data_dir)
    # generate dataset
    data_list = [train_data, val_data] if test_data is None else [train_data, val_data, test_data]
    word_dict, char_dict, vectors = vocab_emb_gen(data_list, emb_path)
    train_set = dataset_gen(train_data, vfeat_lens, word_dict, char_dict, configs.max_pos_len, 'train')
    val_set = dataset_gen(val_data, vfeat_lens, word_dict, char_dict, configs.max_pos_len, 'val')
    test_set = None if test_data is None else dataset_gen(test_data, vfeat_lens, word_dict, char_dict,
                                                          configs.max_pos_len, 'test')
    # save dataset
    n_test = 0 if test_set is None else len(test_set)
    dataset = {'train_set': train_set, 'val_set': val_set, 'test_set': test_set, 'word_dict': word_dict,
               'char_dict': char_dict, 'word_vector': vectors, 'n_train': len(train_set), 'n_val': len(val_set),
               'n_test': n_test, 'n_words': len(word_dict), 'n_chars': len(char_dict)}
    save_pickle(dataset, save_path)
    return dataset


def load_word_and_char_dict(path):
    dataset = load_pickle(path)
    word_dict = dataset['word_dict']
    char_dict = dataset['char_dict']
    vectors = dataset['word_vector']
    return word_dict, char_dict, vectors


def gen_or_load_eval_dataset(configs):
    if not os.path.exists(configs.save_dir):
        os.makedirs(configs.save_dir)
    data_dir = os.path.join('data', 'dataset', configs.task)
    feature_dir = os.path.join('data', 'features', configs.task)

    if configs.suffix is None:
        test_data_save_path = os.path.join(configs.save_dir, '_'.join(
            [configs.task, configs.test_data_file_name, str(configs.max_pos_len)]) +
                                           '.pkl')
        train_data_save_path = os.path.join(configs.save_dir, '_'.join([configs.task, str(configs.max_pos_len)]) +
                                            '.pkl')
    else:
        test_data_save_path = os.path.join(configs.save_dir,
                                           '_'.join([configs.task, 'test' + str(configs.max_pos_len), configs.suffix]) +
                                           '.pkl')
        train_data_save_path = os.path.join(configs.save_dir,
                                            '_'.join([configs.task, str(configs.max_pos_len), configs.suffix]) +
                                            '.pkl')

    if os.path.exists(test_data_save_path):
        dataset = load_pickle(test_data_save_path)
        return dataset

    if os.path.exists(train_data_save_path):
        word_dict, char_dict, vectors = load_word_and_char_dict(train_data_save_path)

    else:
        print("path to processed train data does not exist!")
        sys.exit(0)

    feat_len_path = os.path.join(feature_dir, 'feature_shapes.json')
    # load video feature length
    vfeat_lens = load_json(feat_len_path)
    for vid, vfeat_len in vfeat_lens.items():
        vfeat_lens[vid] = min(configs.max_pos_len, vfeat_len)
    # load data

    if configs.task == 'medvidqa':
        processor = MedVidQAProcessor()
    else:
        raise ValueError('Unknown task {}!!!'.format(configs.task))
    test_data = processor.convert(data_dir, configs.test_data_file_name, eval=True)
    # generate dataset
    test_set = dataset_gen(test_data, vfeat_lens, word_dict, char_dict, configs.max_pos_len, 'test')
    # save dataset
    dataset = {'test_set': test_set,
               'n_test': len(test_set), 'n_words': len(word_dict),
               'n_chars': len(char_dict), 'word_vector': vectors, }
    save_pickle(dataset, test_data_save_path)
    return dataset
