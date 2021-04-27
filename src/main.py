#!/bin/env python3

import pickle
import random
import re
import gc
# import logging
from os import makedirs
from os.path import basename, join, exists
from itertools import tee
from pkg import Preprocessor, LangEnum
from pkg import LDA, Top2VecW
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from pkg import get_files, get_cli_parser, get_text
from tqdm import tqdm
from multiprocessing import Pool

MODEL_CONFIGS = [
    ('lda', {'cls': LDA,
             'workers': 12,
             'num_topics': 10,
             'chunksize': 30,
             'per_word_topics': True,
             'random_state': 42}),
    ('top2vec_doc2vec', {'cls': Top2VecW,
                         'embedding_model': 'doc2vec',
                         'tokenizer': crutch_for_top2vec,
                         'workers': 12}),
    ('top2vec_universal_sentence_encoder', {'cls': Top2VecW,
                                            'embedding_model': 'universal-sentence-encoder-multilingual',
                                            'tokenizer': lambda doc: doc.split(),  # do not preprocess again
                                            'workers': 12}),
    ('top2vec_sbert', {'cls': Top2VecW,
                       'embedding_model': 'distiluse-base-multilingual-cased',
                       'tokenizer': lambda doc: doc.split(),
                       'workers': 12})
]

PREPROC_CONFIGS = [
    ('none', None),
    ('simple', {'tokenize_ents': True, 'workers': 12}),
    ('ner', {'tokenize_ents': False, 'workers': 8})
]

if __name__ == '__main__':
    random.seed(42)

    parser = get_cli_parser()
    args = vars(parser.parse_args())

    dataset_name = basename(args['input_dir'])
    preloaded_path = join(args['input_dir'], dataset_name + '_preloaded')
    makedirs(dataset_name, exist_ok=True)

    print(f"Using {dataset_name} as input document collection")

    print(f"Loading texts")
    if exists(preloaded_path):
        print('Using preloaded text list')
        with open(preloaded_path, 'rb') as f:
            files = pickle.load(f)
    else:
        # List to make processing faster
        with Pool(12) as p:
            files = list(tqdm(p.imap(get_text, get_files(join(args['input_dir'], args['subdir'])))))
        with open(preloaded_path, 'wb') as f:
            pickle.dump(files, f)

    for i, text in enumerate(files):
        r_text, n = re.subn(r'[0-9]+:[0-9]+', '', text)
        files[i] = r_text

    for preproc_name, preproc_cfg in PREPROC_CONFIGS:
        tqdm.write(f'Preprocessing with `{preproc_name}` config')

        if preproc_cfg is not None:
            texts_path = join(args['input_dir'], f'{dataset_name}_preprocessed_{preproc_name}')
            dict_path = join(args['input_dir'], f'{dataset_name}_preprocessed_{preproc_name}_dict')

            if exists(texts_path) and \
                    exists(dict_path):
                print('Using cached result')
                with open(texts_path, 'rb') as f:
                    texts = pickle.load(f)
                dictionary = Dictionary.load(dict_path)

            else:
                print('Running preprocessing from scratch')
                preproc = Preprocessor(language=LangEnum.RU, **preproc_cfg)
                # texts, dictionary = preproc.preprocess_texts(map(get_text, files))
                texts, dictionary = preproc.preprocess_texts(files)
                texts = list(texts)

                with open(texts_path, 'wb') as f:
                    pickle.dump(texts, f)
                with open(dict_path, 'wb') as f:
                    dictionary.save(f)
        else:
            texts = files
            dictionary = Dictionary(documents=[doc.split() for doc in texts])

        for model_name, cfg in MODEL_CONFIGS:
            gc.collect()
            tqdm.write(f'Fitting {model_name}')
            cfg: dict = cfg
            cls = cfg['cls']
            cfg.pop('cls')

            # LDA needs additional arguments 
            # TODO: make uniform API for this
            if cls is LDA:
                doc2bow_corpus = [dictionary.doc2bow(doc)
                                  for doc in map(lambda x: [y.lower_ for y in x],
                                                 texts)]

                model = cls(**cfg, corpus=doc2bow_corpus, id2word=dictionary)
            else:
                model = cls(**cfg, documents=[str(text) for text in texts])

            topics = model.get_topics(num_topics=10)
            topics, t_copy, t_copy_1 = tee(topics, 3)

            ids = list(map(lambda x: x[0], topics))
            words = list(map(lambda x: x[1][0], t_copy))
            scores = list(map(lambda x: x[1][1], t_copy_1))

            if preproc_cfg is not None:
                c_v = CoherenceModel(topics=words,
                                     texts=[[str(token) for token in text] for text in texts],
                                     topn=10,
                                     dictionary=dictionary,
                                     coherence='c_v').get_coherence()
            else:
                c_v = None

            cfg['cls'] = cls

            with open(join(dataset_name,
                           f'{model_name}_{preproc_name}_topics'), 'w') as f:
                for topic in words:
                    f.write(" ".join(topic[:10]) + "\n")
                f.write(f"C_v = {c_v}\n")

            del model
