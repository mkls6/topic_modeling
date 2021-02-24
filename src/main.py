#!/bin/env python3

import pickle
import logging
import datetime as dt
from itertools import tee
from preproc import Preprocessor, LangEnum
from models import LDA, Top2VecW, NMF
from gensim.models.coherencemodel import CoherenceModel

if __name__ == '__main__':
    # parser = get_cli_parser()
    logging.basicConfig(filename=f'{dt.datetime.utcnow()}-log.txt',
                        level=logging.INFO)
    # args = parser.parse_args()

    # files = get_files(args['input_dir'])

    # preprocessor = Preprocessor(language=LangEnum.EN,
    #                             stop_words=None)
    # texts, dictionary = preprocessor.preprocess_texts(
    #     [
    #         'Bleep-bloop, I am a robot!',
    #         'There is a number (12345) and an email (hello-there@box.com).'
    #     ]
    # )
    # print(set(dictionary.values()))

    # Load Interfax subset from Taiga dataset
    logging.info('Loading Interfax texts')
    with open('../loaded_texts', 'rb') as f:
        texts = pickle.load(f)  # Preloaded texts
    texts = texts[:1000]
    # interfax_csv = pd.read_csv(
    #     filepath_or_buffer='/run/media/mk/Media/ml/coursework/datasets'
    #                        '/taiga/news/Interfax/newmetadata.csv',
    #     sep='\t'
    # )

    # Preprocess texts
    logging.info('Preprocessing texts')
    preprocessor = Preprocessor(language=LangEnum.RU)
    doc_iter, dictionary = preprocessor.preprocess_texts(texts)
    preprocessed_texts = list(doc_iter)

    # Free some memory
    del texts

    logging.info('Create gensim corpus object')
    doc2bow_corpus = [dictionary.doc2bow(doc)
                      for doc in map(lambda x: [y.lower_ for y in x],
                                     preprocessed_texts)]

    logging.info('Initialize and fit LDA model')
    lda = LDA(workers=10, corpus=doc2bow_corpus, id2word=dictionary,
              num_topics=10, chunksize=30,
              per_word_topics=True)
    topics = lda.get_topics(num_topics=10)
    topics, t_copy, t_copy_1 = tee(topics, 3)

    ids = list(map(lambda x: x[0], topics))
    words = list(map(lambda x: x[1][0], t_copy))
    scores = list(map(lambda x: x[1][1], t_copy_1))
    # coherence_model = CoherenceModel(topics=words, texts=preprocessed_texts,
    #                                  dictionary=dictionary)
    # logging.info(f'LDA coherence score: {coherence_model.get_coherence()}')

    logging.info(f'Saving LDA top 10 topics')
    with open('./lda-topics', 'w') as f:
        for topic_num, (words, scores) in zip(ids, zip(words, scores)):
            f.write(f'Topic {topic_num}:\n')
            for word, score in zip(words, scores):
                f.write(f'\t{word:10}:{score:6f}\n')

    logging.info('Initialize and fit Top2Vec model')
    top2vec_d2v = Top2VecW(documents=list(map(str, preprocessed_texts)),
                           embedding_model='doc2vec',
                           workers=10)
    topics = top2vec_d2v.get_topics(num_topics=10)
    topics, t_copy, t_copy_1 = tee(topics, 3)

    ids = list(map(lambda x: x[0], topics))
    words = list(map(lambda x: x[1][0], t_copy))
    scores = list(map(lambda x: x[1][1], t_copy_1))
    # coherence_model = CoherenceModel(topics=words,
    #                                  corpus=doc2bow_corpus,
    #                                  texts=preprocessed_texts,
    #                                  dictionary=dictionary)
    # logging.info(f'Top2Vec coherence score:{coherence_model.get_coherence()}')
    logging.info(f'Saving Top2Vec top 10 topics')
    with open('./top2vec-topics', 'w') as f:
        for topic_num, (words, scores) in zip(ids, zip(words, scores)):
            f.write(f'Topic {topic_num}:\n')
            for word, score in zip(words, scores):
                f.write(f'\t{word:10}:{score:6f}\n')

    logging.info('Exiting…')
