# -*- coding: utf-8 -*-
import codecs
import random
import numpy as np
import pickle as pk
from sklearn.metrics import precision_score, recall_score, f1_score
import pdb

def load_w2v(embedding_dim, embedding_dim_pos, train_file_path, embedding_path):
    print('\nload embedding...')

    words = []
    inputFile = codecs.open(train_file_path, 'r', 'utf-8')
    for line in inputFile.readlines():
        line = line.strip().split(',')
        emotion, sen_pos, clause = line[2], line[3], line[-1]
        words.extend([emotion] + [sen_pos] + clause.split())
    words = set(words)  # 所有不重复词的集合
    word_idx = dict((c, k + 1) for k, c in enumerate(words))  # 每个词及词的位置

    w2v = {}
    inputFile = codecs.open(embedding_path, 'r', 'utf-8')
    inputFile.readline()
    for line in inputFile.readlines():
        line = line.strip().split(' ')
        w, ebd = line[0], line[1:]
        w2v[w] = ebd

    embedding = [list(np.zeros(embedding_dim))]
    hit = 0
    for item in words:
        if item in w2v:
            vec = list(map(float, w2v[item]))
            hit += 1
        else:
            vec = list(np.random.rand(embedding_dim) / 5. - 0.1)  # 从均匀分布[-0.1,0.1]中随机取
        embedding.append(vec)
    print('w2v_file: {}\nall_words: {} hit_words: {}'.format(
        embedding_path, len(words), hit))

    embedding_pos = [list(np.zeros(embedding_dim_pos))]
    embedding_pos.extend([
        list(np.random.normal(loc=0.0, scale=0.1, size=embedding_dim_pos))
        for i in range(-68, 34)
    ])

    embedding, embedding_pos = np.array(embedding), np.array(embedding_pos)

    print("embedding.shape: {} embedding_pos.shape: {}".format(
        embedding.shape, embedding_pos.shape))
    print("load embedding done!\n")
    return word_idx, embedding, embedding_pos


def load_data(input_file, word_idx, max_doc_len=25, max_sen_len=45):
    print('load data...')
    print('file_path: {}'.format(input_file))

    y_p, y, x, word_distance, sen_len, doc_len, DGL, special_feature = (
        [], [], [], [], [], [], [], [])

    y_p_doc, y_doc, x_doc, word_distance_doc, sen_len_doc, special_feature_doc = (
        [], [], [], [], [], [])
    DGL_doc, dgl, dgl_id = [], np.zeros((max_doc_len,)), 0
    next_id = 2

    # 定义特殊词汇集合
    #special_words= [u"短短", u"河南", u"甘肃"]
    #special_words = set(['短短', '河南', '葬礼', '丧事'])  # 添加你认为重要的特殊词汇
    special_words = [ u"离世", u"暴富", u"失恋", u"变老", u"遇难", u"失去", u"出事", u"错判", u"当选", u"得病", u"得不到",
                     u"夺冠", u"落败", u"感染", u"去世", u"遭遇", u"患上", u"昏倒", u"获得", u"获奖", u"蒙受", u"溺水", u"失败", u"逝世", u"失去",
                     u"输球", u"享受", u"痛失", u"遭遇", u"遭到", u"遇袭", u"晕倒", u"遇难", u"去世", u"逝世", u"离世", u"早逝", u"辞世", u"冻死",
                     u"绊倒", u"输钱", u"遭受" u"失败"]

    input_file = codecs.open(input_file, 'r', 'utf-8')
    print('sen_pos adding 69')
    print('DGL using -1 as non-cause')
    n_clause, yes_clause, no_clause, n_cut = [0] * 4
    for index, line in enumerate(input_file.readlines()):
        n_clause += 1
        line = line.strip().split(',')
        doc_id, sen_pos, cause, words = int(line[0]), int(line[3]) + 69, line[4], line[5]

        if doc_id == next_id:
            doc_len.append(len(x_doc))
            for j in range(max_doc_len - len(x_doc)):
                y_p_doc.append(np.zeros((102,)))
                y_doc.append(np.zeros((2,)))
                x_doc.append(np.zeros((max_sen_len,)))
                word_distance_doc.append(np.zeros((max_sen_len,)))
                sen_len_doc.append(0)
                DGL_doc.append(np.array(dgl))
                special_feature_doc.append(0)
            y_p.append(y_p_doc)
            y.append(y_doc)
            x.append(x_doc)
            word_distance.append(word_distance_doc)
            sen_len.append(sen_len_doc)
            DGL.append(DGL_doc)
            special_feature.append(special_feature_doc)
            y_p_doc, y_doc, x_doc, word_distance_doc, sen_len_doc, special_feature_doc = (
                [], [], [], [], [], [])
            DGL_doc, dgl, dgl_id = [], np.zeros((max_doc_len,)), 0
            next_id = doc_id + 1

        clause, clause_word_distance = [0] * max_sen_len, [0] * max_sen_len
        clause_contains_special = 0  # 初始化标志位
        for i, word in enumerate(words.split()):
            if i >= max_sen_len:
                n_cut += 1
                break
            clause[i] = int(word_idx.get(word, 0))  # 使用 get 方法防止 KeyError
            clause_word_distance[i] = sen_pos
            if word in special_words:
                clause_contains_special = 1  # 如果包含特殊词，设置标志位

        x_doc.append(np.array(clause))
        word_distance_doc.append(np.array(clause_word_distance))
        sen_len_doc.append(min(len(words.split()), max_sen_len))
        DGL_doc.append(np.array(dgl))
        special_feature_doc.append(clause_contains_special)
        # print ("speccial",special_feature_doc)

        if cause == 'no':
            no_clause += 1
            dgl[dgl_id] = -1
            y_doc.append(np.array([1, 0]))
        else:
            yes_clause += 1
            dgl[dgl_id] = 1
            y_doc.append(np.array([0, 1]))
        dgl_id += 1

        y_p_sen = np.zeros((102,))
        y_p_sen[sen_pos - 1] = 1
        y_p_doc.append(y_p_sen)

    # 处理最后一个文档的数据
    doc_len.append(len(x_doc))
    for j in range(max_doc_len - len(x_doc)):
        y_p_doc.append(np.zeros((102,)))
        y_doc.append(np.zeros((2,)))
        x_doc.append(np.zeros((max_sen_len,)))
        word_distance_doc.append(np.zeros((max_sen_len,)))
        sen_len_doc.append(0)
        DGL_doc.append(np.array(dgl))
        special_feature_doc.append(0)
    y_p.append(y_p_doc)
    y.append(y_doc)
    x.append(x_doc)
    word_distance.append(word_distance_doc)
    sen_len.append(sen_len_doc)
    DGL.append(DGL_doc)
    special_feature.append(special_feature_doc)

    y_p, y, x, sen_len, doc_len, word_distance, DGL, special_feature = map(
        np.array, [y_p, y, x, sen_len, doc_len, word_distance, DGL, special_feature])
    print('y_p.shape {} \ny.shape {} \nx.shape {} \nsen_len.shape {} \n'
          'doc_len.shape {}\nword_distance.shape {}\nDGL.shape {}\n'
          'special_feature.shape {}'.format(
        y_p.shape, y.shape, x.shape, sen_len.shape, doc_len.shape,
        word_distance.shape, DGL.shape, special_feature.shape))
    print('n_clause {}, yes_clause {}, no_clause {}, n_cut {}'.format(
        n_clause, yes_clause, no_clause, n_cut))
    print('load data done!\n')
    return y_p, y, x, sen_len, doc_len, word_distance, DGL, special_feature


def acc_prf(pred_y, true_y, doc_len):
    tmp1, tmp2 = [], []
    for i in range(pred_y.shape[0]):
        for j in range(doc_len[i]):
            tmp1.append(pred_y[i][j])
            tmp2.append(true_y[i][j])
    y_pred, y_true = np.array(tmp1), np.array(tmp2)
    acc = precision_score(y_true, y_pred, average='micro')
    p = precision_score(y_true, y_pred, average='binary')
    r = recall_score(y_true, y_pred, average='binary')
    f1 = f1_score(y_true, y_pred, average='binary')
    return acc, p, r, f1


# encoding: utf-8
# @author: zxding
# email: d.z.x@qq.com
#
# import codecs
# import random
# import numpy as np
# import pickle as pk
# from sklearn.metrics import precision_score, recall_score, f1_score
# import pdb
#
#
# def load_w2v(embedding_dim, embedding_dim_pos, train_file_path, embedding_path):
#     print('\nload embedding...')
#
#     words = []
#     inputFile = codecs.open(train_file_path, 'r', 'utf-8')
#     for line in inputFile.readlines():
#         line = line.strip().split(',')
#         emotion, sen_pos, clause = line[2], line[3], line[-1]
#         words.extend([emotion] + [sen_pos] + clause.split())
#     words = set(words)  # 所有不重复词的集合
#     word_idx = dict((c, k + 1) for k, c in enumerate(words))  # 每个词及词的位置
#
#     w2v = {}
#     inputFile = codecs.open(embedding_path, 'r', 'utf-8')
#     inputFile.readline()
#     for line in inputFile.readlines():
#         line = line.strip().split(' ')
#         w, ebd = line[0], line[1:]
#         w2v[w] = ebd
#
#     embedding = [list(np.zeros(embedding_dim))]
#     hit = 0
#     for item in words:
#         if item in w2v:
#             vec = list(map(float, w2v[item]))
#             hit += 1
#         else:
#             vec = list(np.random.rand(embedding_dim) / 5. - 0.1)  # 从均匀分布[-0.1,0.1]中随机取
#         embedding.append(vec)
#     print('w2v_file: {}\nall_words: {} hit_words: {}'.format(embedding_path, len(words), hit))
#
#     embedding_pos = [list(np.zeros(embedding_dim_pos))]
#     embedding_pos.extend([list(np.random.normal(loc=0.0, scale=0.1, size=embedding_dim_pos)) for i in range(-68, 34)])
#
#     embedding, embedding_pos = np.array(embedding), np.array(embedding_pos)
#
#     print("embedding.shape: {} embedding_pos.shape: {}".format(embedding.shape, embedding_pos.shape))
#     print("load embedding done!\n")
#     return word_idx, embedding, embedding_pos
#
#
# def load_data(input_file, word_idx, max_doc_len=25, max_sen_len=30):
#     print('load data...')
#     print('file_path: {}'.format(input_file))
#
#     y_p, y, x, word_distance, sen_len, doc_len, DGL = [], [], [], [], [], [], []
#
#     y_p_doc, y_doc, x_doc, word_distance_doc, sen_len_doc = [], [], [], [], []
#     DGL_doc, dgl, dgl_id = [], np.zeros((max_doc_len,)), 0
#     next_id = 2
#
#     input_file = codecs.open(input_file, 'r', 'utf-8')
#     print('sen_pos adding 69')
#     print('DGL using -1 as non-cause')
#     n_clause, yes_clause, no_clause, n_cut = [0] * 4
#     for index, line in enumerate(input_file.readlines()):
#         n_clause += 1
#         line = line.strip().split(',')
#         doc_id, sen_pos, cause, words = int(line[0]), int(line[3]) + 69, line[4], line[5]
#         # words = words + ' '+ line[3]
#
#         if doc_id == next_id:  # 数据文件末尾加了一个冗余的文档，会被丢弃
#             doc_len.append(len(x_doc))
#             for j in range(max_doc_len - len(x_doc)):
#                 y_p_doc.append(np.zeros((102,)))
#                 y_doc.append(np.zeros((2,)))
#                 x_doc.append(np.zeros((max_sen_len,)))
#                 word_distance_doc.append(np.zeros((max_sen_len,)))
#                 sen_len_doc.append(0)
#                 DGL_doc.append(np.array(dgl))
#             y_p.append(y_p_doc)
#             y.append(y_doc)
#             x.append(x_doc)
#             word_distance.append(word_distance_doc)
#             sen_len.append(sen_len_doc)
#             DGL.append(DGL_doc)
#             y_p_doc, y_doc, x_doc, word_distance_doc, sen_len_doc = [], [], [], [], []
#             DGL_doc, dgl, dgl_id = [], np.zeros((max_doc_len,)), 0
#             next_id = doc_id + 1
#
#         clause, clause_word_distance = [0] * max_sen_len, [0] * max_sen_len
#         for i, word in enumerate(words.split()):
#             if i >= max_sen_len:
#                 n_cut += 1
#                 break
#             clause[i] = int(word_idx[word])
#             clause_word_distance[i] = sen_pos
#         x_doc.append(np.array(clause))
#         word_distance_doc.append(np.array(clause_word_distance))
#         sen_len_doc.append(min(len(words.split()), max_sen_len))
#         DGL_doc.append(np.array(dgl))
#         if cause == 'no':
#             no_clause += 1
#             dgl[dgl_id] = -1
#             y_doc.append(np.array([1, 0]))
#         else:
#             yes_clause += 1
#             dgl[dgl_id] = 1
#             y_doc.append(np.array([0, 1]))
#         dgl_id += 1
#         y_p_sen = np.zeros((102,))
#         y_p_sen[sen_pos - 1] = 1
#         y_p_doc.append(y_p_sen)
#
#     y_p, y, x, sen_len, doc_len, word_distance, DGL = map(np.array, [y_p, y, x, sen_len, doc_len, word_distance, DGL])
#     print(
#     'y_p.shape {} \ny.shape {} \nx.shape {} \nsen_len.shape {} \ndoc_len.shape {}\nword_distance.shape {}\nDGL.shape {}'.format(
#         y_p.shape, y.shape, x.shape, sen_len.shape, doc_len.shape, word_distance.shape, DGL.shape
#     ))
#     print('n_clause {}, yes_clause {}, no_clause {}, n_cut {}'.format(n_clause, yes_clause, no_clause, n_cut))
#     print('load data done!\n')
#     # pdb.set_trace()
#     return y_p, y, x, sen_len, doc_len, word_distance, DGL
#
#
# def acc_prf(pred_y, true_y, doc_len):
#     tmp1, tmp2 = [], []
#     for i in range(pred_y.shape[0]):
#         for j in range(doc_len[i]):
#             tmp1.append(pred_y[i][j])
#             tmp2.append(true_y[i][j])
#     y_pred, y_true = np.array(tmp1), np.array(tmp2)
#     acc = precision_score(y_true, y_pred, average='micro')
#     p = precision_score(y_true, y_pred, average='binary')
#     r = recall_score(y_true, y_pred, average='binary')
#     f1 = f1_score(y_true, y_pred, average='binary')
#     return acc, p, r, f1


# encoding: utf-8
# @author: zxding
# email: d.z.x@qq.com
#
# import codecs
# import random
# import numpy as np
# import pickle as pk
# from sklearn.metrics import precision_score, recall_score, f1_score
# import pdb
#
#
# def load_w2v(embedding_dim, embedding_dim_pos, train_file_path, embedding_path):
#     print('\nload embedding...')
#
#     words = []
#     inputFile = codecs.open(train_file_path, 'r', 'utf-8')
#     for line in inputFile.readlines():
#         line = line.strip().split(',')
#         emotion, sen_pos, clause = line[2], line[3], line[-1]
#         words.extend([emotion] + [sen_pos] + clause.split())
#     words = set(words)  # 所有不重复词的集合
#     word_idx = dict((c, k + 1) for k, c in enumerate(words))  # 每个词及词的位置
#
#     w2v = {}
#     inputFile = codecs.open(embedding_path, 'r', 'utf-8')
#     inputFile.readline()
#     for line in inputFile.readlines():
#         line = line.strip().split(' ')
#         w, ebd = line[0], line[1:]
#         w2v[w] = ebd
#
#     embedding = [list(np.zeros(embedding_dim))]
#     hit = 0
#     for item in words:
#         if item in w2v:
#             vec = list(map(float, w2v[item]))
#             hit += 1
#         else:
#             vec = list(np.random.rand(embedding_dim) / 5. - 0.1)  # 从均匀分布[-0.1,0.1]中随机取
#         embedding.append(vec)
#     print('w2v_file: {}\nall_words: {} hit_words: {}'.format(embedding_path, len(words), hit))
#
#     embedding_pos = [list(np.zeros(embedding_dim_pos))]
#     embedding_pos.extend([list(np.random.normal(loc=0.0, scale=0.1, size=embedding_dim_pos)) for i in range(-68, 34)])
#
#     embedding, embedding_pos = np.array(embedding), np.array(embedding_pos)
#
#     print("embedding.shape: {} embedding_pos.shape: {}".format(embedding.shape, embedding_pos.shape))
#     print("load embedding done!\n")
#     return word_idx, embedding, embedding_pos
#
#
# def load_data(input_file, word_idx, max_doc_len=25, max_sen_len=30):
#     print('load data...')
#     print('file_path: {}'.format(input_file))
#
#     y_p, y, x, word_distance, sen_len, doc_len, DGL = [], [], [], [], [], [], []
#
#     y_p_doc, y_doc, x_doc, word_distance_doc, sen_len_doc = [], [], [], [], []
#     DGL_doc, dgl, dgl_id = [], np.zeros((max_doc_len,)), 0
#     next_id = 2
#
#     input_file = codecs.open(input_file, 'r', 'utf-8')
#     print('sen_pos adding 69')
#     print('DGL using -1 as non-cause')
#     n_clause, yes_clause, no_clause, n_cut = [0] * 4
#     for index, line in enumerate(input_file.readlines()):
#         n_clause += 1
#         line = line.strip().split(',')
#         doc_id, sen_pos, cause, words = int(line[0]), int(line[3]) + 69, line[4], line[5]
#         # words = words + ' '+ line[3]
#
#         if doc_id == next_id:  # 数据文件末尾加了一个冗余的文档，会被丢弃
#             doc_len.append(len(x_doc))
#             for j in range(max_doc_len - len(x_doc)):
#                 y_p_doc.append(np.zeros((102,)))
#                 y_doc.append(np.zeros((2,)))
#                 x_doc.append(np.zeros((max_sen_len,)))
#                 word_distance_doc.append(np.zeros((max_sen_len,)))
#                 sen_len_doc.append(0)
#                 DGL_doc.append(np.array(dgl))
#             y_p.append(y_p_doc)
#             y.append(y_doc)
#             x.append(x_doc)
#             word_distance.append(word_distance_doc)
#             sen_len.append(sen_len_doc)
#             DGL.append(DGL_doc)
#             y_p_doc, y_doc, x_doc, word_distance_doc, sen_len_doc = [], [], [], [], []
#             DGL_doc, dgl, dgl_id = [], np.zeros((max_doc_len,)), 0
#             next_id = doc_id + 1
#
#         clause, clause_word_distance = [0] * max_sen_len, [0] * max_sen_len
#         for i, word in enumerate(words.split()):
#             if i >= max_sen_len:
#                 n_cut += 1
#                 break
#             clause[i] = int(word_idx[word])
#             clause_word_distance[i] = sen_pos
#         x_doc.append(np.array(clause))
#         word_distance_doc.append(np.array(clause_word_distance))
#         sen_len_doc.append(min(len(words.split()), max_sen_len))
#         DGL_doc.append(np.array(dgl))
#         if cause == 'no':
#             no_clause += 1
#             dgl[dgl_id] = -1
#             y_doc.append(np.array([1, 0]))
#         else:
#             yes_clause += 1
#             dgl[dgl_id] = 1
#             y_doc.append(np.array([0, 1]))
#         dgl_id += 1
#         y_p_sen = np.zeros((102,))
#         y_p_sen[sen_pos - 1] = 1
#         y_p_doc.append(y_p_sen)
#
#     y_p, y, x, sen_len, doc_len, word_distance, DGL = map(np.array, [y_p, y, x, sen_len, doc_len, word_distance, DGL])
#     print(
#     'y_p.shape {} \ny.shape {} \nx.shape {} \nsen_len.shape {} \ndoc_len.shape {}\nword_distance.shape {}\nDGL.shape {}'.format(
#         y_p.shape, y.shape, x.shape, sen_len.shape, doc_len.shape, word_distance.shape, DGL.shape
#     ))
#     print('n_clause {}, yes_clause {}, no_clause {}, n_cut {}'.format(n_clause, yes_clause, no_clause, n_cut))
#     print('load data done!\n')
#     # pdb.set_trace()
#     return y_p, y, x, sen_len, doc_len, word_distance, DGL
#
#
# def acc_prf(pred_y, true_y, doc_len):
#     tmp1, tmp2 = [], []
#     for i in range(pred_y.shape[0]):
#         for j in range(doc_len[i]):
#             tmp1.append(pred_y[i][j])
#             tmp2.append(true_y[i][j])
#     y_pred, y_true = np.array(tmp1), np.array(tmp2)
#     acc = precision_score(y_true, y_pred, average='micro')
#     p = precision_score(y_true, y_pred, average='binary')
#     r = recall_score(y_true, y_pred, average='binary')
#     f1 = f1_score(y_true, y_pred, average='binary')
#     return acc, p, r, f1
#
#

