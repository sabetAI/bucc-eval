# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import io
from logging import getLogger
import numpy as np
import torch

from src.utils import bow_idf, get_nn_avg_dist
from collections import namedtuple
from ipdb import set_trace
from tqdm import tqdm


BUCC_DIR = 'data/crosslingual/bucc2018'


logger = getLogger()


def load_bucc_data(lg1, lg2, split, n_max=1e10, lower=True, full=False):
    """
    Load data parallel sentences
    """
    # if not (os.path.isfile(os.path.join(BUCC_DIR, 'bucc2018.%s-%s.%s' % (lg1, lg2, lg1))) or
    #         os.path.isfile(os.path.join(BUCC_DIR, 'bucc2018.%s-%s.%s' % (lg2, lg1, lg1)))):
    #     return None

    # if os.path.isfile(os.path.join(BUCC_DIR, 'bucc2018.%s-%s.%s' % (lg2, lg1, lg1))):
    #     lg1, lg2 = lg2, lg1

    # load sentences
    data = {lg1: [], lg2: []}
    for lg in [lg1, lg2]:
        if full:
            fname = os.path.join(BUCC_DIR, 'bucc2018.%s-%s.%s.%s' % (lg1, lg2,
                                                                     split, lg))
        else:
            fname = os.path.join(BUCC_DIR, 'bucc2018.%s-%s.%s' % (lg1, lg2, lg))

        with io.open(fname, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= n_max:
                    break
                line = line.lower() if lower else line
                data[lg].append(line.rstrip().split())

    # get only unique sentences for each language
    if not full:
        assert len(data[lg1]) == len(data[lg2])
    data[lg1] = np.array(data[lg1])
    data[lg2] = np.array(data[lg2])
    data[lg1] = list(zip(range(len(data[lg1])+1), data[lg1]))
    data[lg2] = list(zip(range(len(data[lg2])+1), data[lg2]))

    logger.info("Loaded bucc2018 %s.%s-%s (%i sentences)." % (split, lg1, lg2, len(data[lg1])))
    return data

def load_bucc_labels(lg1, lg2, split, n_max=1e10, lower=True, full=False):
    """
    Load data parallel sentences
    """
    if split == 'test':
        return None
    if not os.path.isfile(os.path.join(BUCC_DIR, 'bucc2018.%s-%s.%s.gold' % (lg1, lg2, split))):
        return None

    # load sentences
    labels = []
    fname = os.path.join(BUCC_DIR, 'bucc2018.%s-%s.%s.gold' % (lg1, lg2, split))

    with io.open(fname, 'r') as f:
        for ids in f:
            ids = ids.split('\t')
            labels.append([int(ids[0]), int(ids[1])])

    # shuffle sentences
    labels = np.array(labels)

    logger.info("Loaded bucc %s labels %s-%s (%i sentences)." % (split, lg1, lg2, len(labels)))
    return labels


def get_sent_translation_accuracy(data, labels, lg1, word2id1, emb1, lg2, word2id2, emb2,
                                  method, idf, test):

    """
    Given parallel sentences from Europarl, evaluate the
    sentence translation accuracy using the precision@k.
    """
    # get word vectors dictionaries
    emb1 = emb1.cpu().numpy()
    emb2 = emb2.cpu().numpy()
    word_vec1 = dict([(w, emb1[word2id1[w]]) for w in word2id1])
    word_vec2 = dict([(w, emb2[word2id2[w]]) for w in word2id2])
    word_vect = {lg1: word_vec1, lg2: word_vec2}
    lg_keys = lg2
    lg_query = lg1

    # get n_keys pairs of sentences
    src_keys = torch.LongTensor(labels[:,0]) if labels else torch.arange(len(data[lg1]))
    tgt_keys = torch.LongTensor(labels[:,1]) if labels else torch.arange(len(data[lg2]))
    keys = data[lg_keys]
    key_ids, keys = bow_idf(keys, word_vect[lg_keys], idf_dict=idf[lg_keys])

    # get n_queries query pairs from these n_keys pairs
    rng = np.random.RandomState(1234)
    queries = [data[lg_query][i.item()] for i in src_keys]
    query_ids, queries = bow_idf(queries, word_vect[lg_query], idf_dict=idf[lg_query])

    # normalize embeddings
    queries = torch.from_numpy(queries).float()
    queries = queries / queries.norm(2, 1, keepdim=True).expand_as(queries)
    keys = torch.from_numpy(keys).float()
    keys = keys / keys.norm(2, 1, keepdim=True).expand_as(keys)


    # nearest neighbors
    if method == 'nn':
        top1 = top1_scores(queries, keys, 3000)

    # contextual dissimilarity measure
    elif method.startswith('csls_knn_'):
        knn = method[len('csls_knn_'):]
        assert knn.isdigit()
        knn = int(knn)
        # average distances to k nearest neighbors
        knn = method[len('csls_knn_'):]
        assert knn.isdigit()
        knn = int(knn)
        average_dist_keys = torch.from_numpy(get_nn_avg_dist(queries, keys, knn))
        average_dist_queries = torch.from_numpy(get_nn_avg_dist(keys, queries, knn))
        # scores
        scores = keys.mm(queries.transpose(0, 1)).transpose(0, 1)
        scores.mul_(2)
        scores.sub_(average_dist_queries[:, None].float() + average_dist_keys[None, :].float())
        scores = scores.cpu()

    results = []
    top_matches = scores.topk(10, 1, True)[1]
    predictions = top_matches[:,0]

    if not test:
        for k in [1, 5, 10]:
            top_k_matches = (top_matches[:, :k] == tgt_keys[:, None]).sum(1)
            precision_at_k = 100 * top_k_matches.float().numpy().mean()
            logger.info("%i queries (%s) - %s - Precision at k = %i: %f" %
                        (len(top_k_matches), lg_query.upper(), method, k, precision_at_k))
            results.append(('sent-precision_at_%i' % k, precision_at_k))

    return predictions, results

def top1_scores(queries, keys, batch_size):
    all_scores = []
    all_idx = []
    for q in tqdm(queries.split(batch_size)):
        scores, idx = keys.mm(q.t()).t().topk(1)
        all_scores.append(scores)
        all_idx.append(idx)
    all_scores = torch.cat(all_scores)
    all_idx = torch.cat(all_idx)
    return (all_scores, all_idx)
