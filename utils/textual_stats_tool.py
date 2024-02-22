
import glob
import json
import os
import string
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import nltk
from nltk.corpus import wordnet as wn
from wordcloud import STOPWORDS, WordCloud

from type.TrackingType import TrackingQuery


def preprocess_text(text: str | List[str]) -> str:
    """
    Preprocess text by removing punctuation and making it lowercase
    :param text: text to preprocess
    :return: preprocessed text
    """
    if isinstance(text, list):
        text = ' '.join(text)
    if not isinstance(text, str):
        raise ValueError('Text is not a string: {}'.format(text))
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    return text


def unique_word_count(gt_text_queries: List[TrackingQuery], field: str = 'text') -> int:
    """
    Count the number of words in the text queries
    :param gt_text_queries: list of text queries
    :param field: field to count the words from
    :return: number of words
    """
    count = 0
    all_words = []
    for query in gt_text_queries:
        if (field == 'type'):
            if query.is_eval == False:
                continue
        if not hasattr(query, field):
            raise ValueError(
                'Field {} not found in query {}'.format(field, query))

        preprocess_text_field = preprocess_text(getattr(query, field))
        preprocess_text_field = preprocess_text_field.translate(
            str.maketrans('', '', string.punctuation))
        preprocess_text_field = preprocess_text_field.lower()
        all_words.extend(preprocess_text_field.split())

    summary, repeat = np.unique(all_words, return_counts=True)
    count = len(summary)
    return count, summary, repeat


def build_word_cloud(gt_text_queries: List[TrackingQuery], filter_tags: List[str] = ['NN'], field: str = 'text') -> WordCloud:
    """
    Build word cloud from text queries
    :param gt_text_queries: list of text queries
    :param filter_tags: list of tags to filter the words from (if empty, no filtering is done, else only words with the specified tags are kept)
    :return: None
    """
    stopwords = set(STOPWORDS)

    filter_words = []
    for query in gt_text_queries:
        # Get subject in caption using NLTK wn
        all_words = preprocess_text(getattr(query, field))
        # Get 3 words before and after the subject
        tokens = nltk.word_tokenize(all_words)
        pos_tags = nltk.pos_tag(tokens)
        for word, pos in pos_tags:
            if len(filter_tags) == 0 or pos in filter_tags:
                filter_words.append(word)

    return WordCloud(stopwords=stopwords, collocations=False, background_color='white').generate(' '.join(filter_words))


def count_avg_sentence_length(gt_text_queries: List[TrackingQuery], field: str = 'text') -> int:
    """
    Count the number of words in the text queries
    :param gt_text_queries: list of text queries
    :param field: field to count the words from
    :return: number of words
    """
    sentence_lens = []
    for query in gt_text_queries:
        if not hasattr(query, field):
            raise ValueError(
                'Field {} not found in query {}'.format(field, query))

        preprocess_text_field = preprocess_text(getattr(query, field))
        preprocess_text_field = preprocess_text_field.translate(
            str.maketrans('', '', string.punctuation))
        preprocess_text_field = preprocess_text_field.lower()
        sentence_lens.append(len(preprocess_text_field.split(' ')))

    return np.sum(sentence_lens) / len(sentence_lens)
