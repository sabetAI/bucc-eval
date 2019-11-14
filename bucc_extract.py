import os
from math import ceil
from argparse import ArgumentParser
from collections import defaultdict
import spacy
from multiprocessing import Pool
import numpy as np
from tqdm import tqdm
from pdb import set_trace
from spacy.lang.ru import Russian
from spacy_russian_tokenizer import RussianTokenizer, MERGE_PATTERNS

def split(lst, nsplit):
    size = ceil(len(lst)/nsplit)
    print(size)
    splits = list(range(0, len(lst), size))
    lsts = [lst[i:i+size] for i,j in zip(splits, splits[1:] + [None])] 
    return [l for l in lsts if len(l)]

def join(lsts):
    joined = []
    for lst in lsts:
        joined += lst
    return joined


def parallel_tokenize(sentences, nprocs):
    splits = split(sentences, nprocs)
    pool = Pool(nprocs)
    joined = join(pool.map(tokenize, splits))

    return joined


def tokenize(sentences):
    sentences = list(sentences)
    tokenized_sents = []
    tokenizer = l1nlp if d.split('.')[1] != 'en' else l2nlp

    for sentence in tqdm(sentences):
        tokens = [word.text for word in tokenizer(sentence)]
        tokenized = ' '.join(tokens)
        tokenized_sents.append(tokenized)

    return tokenized_sents


argparser = ArgumentParser()
argparser.add_argument('dir')
argparser.add_argument('nprocs', type=int)

args = argparser.parse_args()

files = os.listdir(args.dir)

lang1, lang2 = args.dir.split('-')

lang1_sents = []
lang2_sents = []

datasets = {}
idmap = {}

if lang1 == 'fr':
    l1nlp = spacy.load(lang1 + "_core_news_sm")
if lang1 == 'de':
    l1nlp = spacy.load(lang1 + "_core_news_sm")
if lang1 == 'ru':
    l1nlp = Russian()
    tokenizer = RussianTokenizer(l1nlp, MERGE_PATTERNS)
    l1nlp.add_pipe(tokenizer, name='russian_tokenizer')
if lang2 == 'en':
    l2nlp = spacy.load(lang2 + "_core_web_sm")

for fname in files:
    if 'gold' not in fname:
        fsplit = fname.split('.', 1)[1]
        datasets[fsplit] = {}
        with open(os.path.join(args.dir, fname), encoding='utf-8') as fp:
            for line in fp:
                line_id, text = line.split('\t', 1)
                line_id = int(line_id.split('-')[1])
                datasets[fsplit][line_id] = text.strip()

    else:
        with open(os.path.join(args.dir, fname), encoding='utf-8') as fp:
            for line in fp:
                lang1_id, lang2_id = line.split('\t')
                lang1_id = int(lang1_id.split('-')[1].strip())
                lang2_id = int(lang2_id.split('-')[1].strip())

                idmap[lang1_id] = lang2_id

for d, sentences in datasets.items():
    with open(os.path.join(args.dir, 'bucc2018.'+ args.dir + '.' + d), 'w',
              encoding='utf-8') as fp:
        sentences = list(sentences.values())
        tokenized = parallel_tokenize(sentences, args.nprocs)
        for sentence in tokenized:
            fp.write(sentence + '\n')

with open(os.path.join(args.dir, 'bucc2018.' + args.dir + '.gold'), 'w') as fp:
    for l1key, l2key in idmap.items():
        fp.write(str(l1key) + '\t' + str(l2key) + '\n')

