# -*- encoding:utf-8 -*-
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
    # special_words = [ u"离世", u"暴富", u"失恋", u"变老", u"遇难", u"失去", u"出事", u"错判", u"当选", u"得病", u"得不到",
    #                  u"夺冠", u"落败", u"感染", u"去世", u"遭遇", u"患上", u"昏倒", u"获得", u"获奖", u"蒙受", u"溺水", u"失败", u"逝世", u"失去",
    #                  u"输球", u"享受", u"痛失", u"遭遇", u"遭到", u"遇袭", u"晕倒", u"遇难", u"去世", u"逝世", u"离世", u"早逝", u"辞世", u"冻死",
    #                  u"绊倒", u"输钱", u"遭受" u"失败"]
    special_words = [u"百战百胜", u"摆脱",u"被抓获",u"享受",u"败给",u"陷入",u"饱受",u"被禁锢",u"离世",u"去世",u"死亡",u"失去",u"暴富",u"崩溃",u"成功",u"当上",u"失恋",u"变成熟",u"落后",u"变坏",u"被视作",u"变老",u"遍体鳞伤",
u"遭到",u"病倒",u"博得",u"遇难",u"受伤",u"被困",u"实现",u"取得",u"受到",u"完成",u"丢失",u"得到",u"获得",u"失利",u"战胜",u"失败",u"承受",u"吃饱",u"吃多",u"吃亏",u"出错",u"出名",u"出事",u"被杀害",
u"错过",u"被错判",u"遭遇",u"错判",u"达成",u"打错",u"被打",u"被骂",u"击败",u"夺冠",u"学会",u"打赢",u"输",u"当选",u"被控",u"得病",u"得不到",u"忍受",u"被打瞎",u"过世",u"被弃",u"夺取",u"犯规",u"得罪",
u"跌倒",u"丢",u"被盗",u"夺得",u"落败",u"躲过",u"发错",u"反败为胜",u"犯错误",u"赶上",u"感染",u"搞乱",u"功成名就",u"管理好",u"赢得",u"被划破",u"患病",u"患有",u"患上",u"被忽视",u"被查出",u"遭受",u"被诊断",
u"毁容",u"昏倒",u"被认可",u"遭拒绝",u"获奖",u"被评为",u"获救",u"获胜",u"被批斗",u"家破人亡",u"解脱",u"救活",u"看懂",u"看清楚",u"亏损",u"立功",u"领先",u"流产",u"落选",u"蒙受",u"蒙羞",u"迷路",u"被免于",u"面临",
u"摸清",u"溺水",u"弄巧成拙",u"弄清",u"赔本",u"碰壁",u"取胜",u"染上",u"惹恼",u"被暗杀",u"被拔下",u"荣登",u"入选",u"上当",u"身陷",u"输给",u"生病",u"被剥夺",u"被打晕",u"走错",u"受罪",u"遭",u"战死",u"失眠",
u"赚钱",u"失窃",u"逝世",u"失业",u"被分配",u"摔倒",u"被摔倒",u"杀害",u"被仇恨",u"瘫痪",u"被绑架",u"听懂",u"痛失",u"脱险",u"被启用",u"被通知",u"误伤",u"被烧毁",u"被抛弃",u"被污染",u"享受到",u"被揭发",
u"养活",u"一事无成",u"遗失",u"赢",u"被赋予",u"赢利",u"用错",u"遇袭",u"晕倒",u"被罚",u"被偷走",u"被砍杀",u"被色诱",u"被指控",u"沾染",u"被掩埋",u"被录取",u"中奖",u"赚到",u"赚取",u"被抢劫",u"走散",u"还清",u"当好",
u"受灾",u"灾害",u"病逝",u"被善待",u"英年早逝",u"溘然长逝",u"早逝",u"受益",u"仙逝",u"与世长辞",u"离开人世",u"辞世",u"被埋",u"被",u"截肢",u"受损",u"被侵害",u"受苦",u"被伤",u"受困",u"受刑",u"被捆",u"被毁",u"被攻击",
u"被贬低",u"被炒",u"被压",u"被击垮",u"被奸杀",u"被冻死",u"被冲垮",u"被击碎",u"被偷",u"被剪掉",u"被卷入",u"被发现",u"被吊打",u"被炸断",u"被碾死",u"被丢弃",u"被列为",u"被公认为",u"被损害",u"被罚款",
u"被感染",u"被接受",u"被选派",u"被当作",u"被扯坏",u"被判断",u"被罚站",u"被抱",u"被窃",u"被吓倒",u"被请出",u"被撞见",u"被顶撞",u"被追尾",u"被砍伐",u"被质疑",u"被吞掉",u"被糟蹋",u"被软禁",u"被忽悠",u"被问到",u"被索要",
u"被霸占",u"被拿走",u"被折腾",u"被出卖",u"被遗弃",u"被强制",u"被关进",u"被证实",u"被知道",u"被看作",u"被收取",u"被贴",u"被报复",u"被困扰",u"被抓",u"被偷吃",u"被袭",u"被问倒",u"被听到",u"被抱错",u"被点名",
u"被咬伤",u"被责怪",u"被绊倒",u"被罢免",u"被炸",u"被轰炸",u"被说",u"被批准",u"被提拔",u"被信任",u"被赶",u"被没收",u"被踢",u"被问",u"被骗",u"被逮捕",u"被禁止",u"被释放",u"被欺骗",u"被拒绝",u"被排除",u"被杀",u"被阻止",
u"被要求",u"被追问",u"被愚弄",u"被辞退",u"被开除",u"被淘汰",u"被审讯",u"被抢",u"被纠缠",u"被束缚",u"被限制",u"被贩卖",u"被折磨",u"被掌握",u"被强迫",u"被执行",u"被讥笑",u"被冻",u"被包围",u"被碰",
u"被支配",u"被追究",u"被讹诈",u"被冤枉",u"被拐骗",u"被处罚",u"被调查",u"被影响",u"被敲诈"]

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
