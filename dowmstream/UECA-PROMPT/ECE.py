import argparse
import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForMaskedLM
import time
import numpy as np

"""setting argparse"""
parser = argparse.ArgumentParser(description='Training')

"""model struct"""
parser.add_argument('--n_hidden', type=int, default=100, help='number of hidden unit')
parser.add_argument('--n_class', type=int, default=2, help='number of distinct class')
parser.add_argument('--window_size', type=int, default=2, help='size of the emotion cause pair window')
parser.add_argument('--feature_layer', type=int, default=3, help='number of layer iterations')
parser.add_argument('--log_file_name', type=str, default='log', help='name of log file')
parser.add_argument('--model_type', type=str, default='ISML', help='type of model')
"""training"""
parser.add_argument('--training_iter', type=int, default=20, help='number of train iterator')
parser.add_argument('--scope', type=str, default='Ind_BiLSTM', help='scope')
parser.add_argument('--batch_size', type=int, default=8, help='number of example per batch')
parser.add_argument('--learning_rate', type=float, default=0.00001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.01, help='weight decay for bert')
parser.add_argument('--usegpu', type=str, default='True', help='gpu')
"""other"""
parser.add_argument('--test_only', type=str, default='True', help='no training')
parser.add_argument('--checkpoint', type=str, default='True', help='load checkpoint')
parser.add_argument('--checkpointpath', type=str, default='prompt_ECE', help='path to load checkpoint')
parser.add_argument('--savecheckpoint', type=str, default='True', help='save checkpoint')
parser.add_argument('--save_path', type=str, default='prompt_ECE', help='path to save checkpoint')
parser.add_argument('--device', type=str, default='3', help='device id')
parser.add_argument('--dataset', type=str, default='data_combine_ECE/', help='path for dataset')
parser.add_argument('--mode', type=str, default='indicator', help='baseline/indicator/event')
parser.add_argument('--event_file', type=str, default='D4_指示词主题映射.txt', help='事件库文件路径')

opt = parser.parse_args()

# 字符串布尔参数转 bool
def _str2bool(v):
    return str(v).lower() in ('true', '1', 'yes', 't')

opt.usegpu = _str2bool(opt.usegpu)
opt.test_only = _str2bool(opt.test_only)
opt.checkpoint = _str2bool(opt.checkpoint)
opt.savecheckpoint = _str2bool(opt.savecheckpoint)

os.environ["CUDA_VISIBLE_DEVICES"] = opt.device

if opt.usegpu and torch.cuda.is_available():
    use_gpu = True
else:
    use_gpu = False  # 添加对GPU不可用时的处理

def print_time():
    print('\n----------{}----------'.format(time.strftime("%Y-%m-%d %X", time.localtime())))

class MyDataset(Dataset):
    def __init__(self, input_file, test=False, tokenizer=None):
        print('load data_file: {}'.format(input_file))
        self.x_bert, self.y_bert, self.label, self.mask_label = [], [], [], []
        self.gt_cause = []
        self.ECE = []
        self.doc_id = []
        self.special_word_ids = []
        self.test = test
        self.n_cut = 0
        self.tokenizer = tokenizer

        inputFile = open(input_file, 'r', encoding='utf-8')

        # 事件库(方法2: 指示词+主题词匹配), 从文件加载
        self.event_lib = {}
        if opt.mode == 'event' and os.path.exists(opt.event_file):
            with open(opt.event_file, 'r', encoding='utf-8') as ef:
                for l in ef:
                    l = l.rstrip('\n')
                    if not l:
                        continue
                    parts = l.split('\t')
                    if len(parts) != 2:
                        continue
                    ind, theme = parts[0], parts[1]
                    if len(theme) >= 2:
                        self.event_lib.setdefault(ind, []).append(theme[:2])
            n = sum(len(v) for v in self.event_lib.values())
            print('加载事件库(指示词->主题): %d 个指示词, %d 个主题' % (len(self.event_lib), n))
        else:
            self.event_lib = {}

        self.special_words = [u"百战百胜", u"摆脱",u"被抓获",u"享受",u"败给",u"陷入",u"饱受",u"被禁锢",u"离世",u"去世",u"死亡",u"失去",u"暴富",u"崩溃",u"成功",u"当上",u"失恋",u"变成熟",u"落后",u"变坏",u"被视作",u"变老",u"遍体鳞伤",
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

        while True:
            line = inputFile.readline()
            if line == '':
                break

            line = line.strip().split()
            self.doc_id.append(line[0])
            d_len = int(line[1])

            pairs = eval('[' + inputFile.readline().strip() + ']')
            pos, cause = zip(*pairs)

            part_sentence = []
            for _ in range(d_len):
                words = inputFile.readline().strip().split(',')[-1]
                part_sentence.append(words)

            self.gt_cause.append(len(set(cause)))

            # ========== 构造文本 ==========
            full_document = ""
            mask_full_document = ""
            mask_label_full_document = ""
            ECE_document = ""
            mask_label_ECE_document = ""

            special_word_flags = []

            for i in range(1, d_len + 1):
                words = part_sentence[i - 1]

                # special word flag（根据 mode 决定匹配方式）
                if opt.mode == 'baseline':
                    special_word_flags.append(0)
                elif opt.mode == 'indicator':
                    if any(sw in words for sw in self.special_words):
                        special_word_flags.append(1)
                    else:
                        special_word_flags.append(0)
                else:  # event: 方法2(指示词+主题词双匹配)
                    hit = 0
                    for ind, themes in self.event_lib.items():
                        if ind in words:
                            for th in themes:
                                if th in words:
                                    hit = 1
                                    break
                        if hit:
                            break
                    special_word_flags.append(hit)

                full_document += f" {i} {words}"
                mask_full_document += f" {i} {words}"
                mask_label_full_document += f" [MASK] {words}"
                ECE_document += f" {i} {words}"
                mask_label_ECE_document += f" [MASK] {words}"

                if i in pos:
                    full_document += "是 "
                    ECE_document += "是 "
                    mask_label_ECE_document += "是 "
                    if i in cause:
                        full_document += f" {pos[cause.index(i)]} "
                    else:
                        full_document += " 无 "
                else:
                    full_document += "非 "
                    ECE_document += "非 "
                    mask_label_ECE_document += "非 "
                    if i in cause:
                        full_document += f" {pos[cause.index(i)]} "
                    else:
                        full_document += " 无 "

                full_document += "[SEP]"
                mask_full_document += " [MASK] [MASK] [MASK] [SEP]"
                mask_label_full_document += " [MASK] [MASK] [MASK] [SEP]"
                ECE_document += " [MASK] [MASK] [SEP]"
                mask_label_ECE_document += " [MASK] [MASK] [SEP]"

            # ========== tokenize ==========
            def encode(x):
                return self.tokenizer.encode_plus(
                    x,
                    return_tensors="pt",
                    max_length=512,
                    truncation=True,
                    padding='max_length'
                )['input_ids'][0]

            mask_full_document_ids = encode(mask_full_document)
            full_document_ids = encode(full_document)
            mask_label_full_document_ids = encode(mask_label_full_document)
            ECE_document_ids = encode(ECE_document)
            mask_label_ECE_document_ids = encode(mask_label_ECE_document)

            labels = full_document_ids.masked_fill(ECE_document_ids != 103, -100)
            mask_labels = full_document_ids.masked_fill(mask_label_full_document_ids != 103, -100)

            # ========== ✔ 修复点：动态 offset，不再 +4 ==========
            special_word_ids = torch.zeros_like(full_document_ids)

            sentence_start_positions = []
            sentence_end_positions = []

            current_position = 1  # CLS

            for i in range(d_len):
                sentence = f" {i+1} {part_sentence[i]}"

                encoded_sentence = self.tokenizer.encode(
                    sentence,
                    add_special_tokens=False
                )

                start = current_position
                end = current_position + len(encoded_sentence)

                sentence_start_positions.append(start)
                sentence_end_positions.append(end)

                current_position = end

            # mark special word spans
            for idx, flag in enumerate(special_word_flags):
                if flag == 1:
                    start = sentence_start_positions[idx]
                    end = sentence_end_positions[idx]
                    end = min(end, 512)
                    if start < 512:
                        special_word_ids[start:end] = 1

            # ========== save ==========
            self.special_word_ids.append(special_word_ids.numpy())
            self.x_bert.append(mask_label_full_document_ids.numpy())
            self.y_bert.append(full_document_ids.numpy())
            self.label.append(labels.numpy())
            self.mask_label.append(mask_labels.numpy())
            self.ECE.append(ECE_document_ids.numpy())

        # convert to numpy
        self.x_bert = np.array(self.x_bert)
        self.y_bert = np.array(self.y_bert)
        self.label = np.array(self.label)
        self.mask_label = np.array(self.mask_label)
        self.ECE = np.array(self.ECE)
        self.special_word_ids = np.array(self.special_word_ids)
        self.gt_cause = np.array(self.gt_cause)

        for var in ['self.x_bert', 'self.y_bert', 'self.label',
                    'self.mask_label', 'self.ECE', 'self.gt_cause', 'self.special_word_ids']:
            print(f'{var}.shape {eval(var).shape}')

        self.index = list(range(len(self.x_bert)))

    def __getitem__(self, index):
        index = self.index[index]
        return [
            self.x_bert[index],
            self.y_bert[index],
            self.label[index],
            self.mask_label[index],
            self.ECE[index],
            self.gt_cause[index],
            self.special_word_ids[index]
        ]

    def __len__(self):
        return len(self.x_bert)
class prompt_bert(torch.nn.Module):
    def __init__(self, bert_path='./bert-base-chinese'):
        super(prompt_bert, self).__init__()
        self.bert = BertForMaskedLM.from_pretrained(bert_path)
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        self.bert.resize_token_embeddings(len(self.tokenizer))
        # 新增的嵌入层，用于特殊词特征
        self.special_word_embeddings = torch.nn.Embedding(2, self.bert.config.hidden_size)

    def forward(self, input_ids, attention_mask=None, special_word_ids=None, labels=None):
        # 获取原始的词嵌入
        outputs = self.bert.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        sequence_output = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]

        # 获取特殊词的嵌入
        if special_word_ids is not None:
            # print("special_word_id",special_word_ids)
            special_word_embeddings = self.special_word_embeddings(special_word_ids)
            # print("special_word_embeddings",special_word_embeddings)
            # 将特殊词嵌入与词嵌入相加
            sequence_output += special_word_embeddings

        # 将融合后的嵌入传递给语言模型头
        prediction_scores = self.bert.cls(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            # 忽略label中的-100（在BertForMaskedLM中也是这样处理的）
            loss = loss_fct(prediction_scores.view(-1, self.bert.config.vocab_size), labels.view(-1))

        return loss, prediction_scores

def print_training_info():
    print('\n\n>>>>>>>>>>>>>>>>>>>>TRAINING INFO:\n')
    print('batch-{}, lr-{}'.format(
        opt.batch_size, opt.learning_rate))
    print('training_iter-{}\n'.format(opt.training_iter))

def prf_prompt(logits, labels, x_bert, gt_cause):
    cause_gt = torch.sum(gt_cause)
    cause_pre = 0
    cause_acc = 0
    for i in range(labels.shape[0]):
        count_mask = -1
        j = 0
        while j < 512:
            if x_bert[i][j] == 103:
                count_mask += 1
                count_mask = count_mask % 2
                if count_mask == 0:
                    if torch.argmax(logits[i][j]) == 3221:
                        if labels[i][j] == 3221:
                            cause_acc += 1
                    if torch.argmax(logits[i][j]) == 3221:
                        cause_pre += 1

                if count_mask == 1:
                    pass
                j = j + 1
            else:
                j = j + 1
    p_cause = cause_acc / (cause_pre + 1e-8)
    r_cause = cause_acc / (cause_gt + 1e-8)
    f_cause = 2 * p_cause * r_cause / (p_cause + r_cause + 1e-8)
    print('cause_gt {}'.format(cause_gt))
    return p_cause, r_cause, f_cause

def run():
    if opt.log_file_name:
        save_path = opt.save_path

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # sys.stdout = open(save_path + '/' + opt.log_file_name, 'w')

    print_time()
    bert_path = './bert-base-chinese'
    tokenizer = BertTokenizer.from_pretrained(bert_path)

    # train
    print_training_info()  # 输出训练的超参数信息
    max_result_cause_f, max_result_cause_r, max_result_cause_p = [], [], []
    for fold in range(1, 11):
        # model
        print('build model..')
        model = prompt_bert(bert_path)
        print('build model end...')
        if opt.checkpoint:
            # 加载官方 checkpoint: 只灌 bert 权重, 保留 special_word_embeddings 增强层
            ckpt_path = opt.checkpointpath + '/fold{}.pth'.format(fold)
            ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
            if hasattr(ckpt, 'state_dict'):
                ckpt_sd = ckpt.state_dict()
            else:
                ckpt_sd = ckpt
            model_sd = model.state_dict()
            # 只加载 bert.* 的键(增强层 special_word_embeddings 保持随机初始化)
            for k, v in ckpt_sd.items():
                if k.startswith('bert.') and k in model_sd:
                    model_sd[k] = v
            model.load_state_dict(model_sd, strict=False)
            print('已加载官方 bert 权重(保留事件增强层)')
        if use_gpu:
            model = model.cuda()

        train_file_name = 'fold{}_train.txt'.format(fold)
        test_file_name = 'fold{}_test.txt'.format(fold)
        print('############# fold {} begin ###############'.format(fold))
        train = opt.dataset + train_file_name
        test = opt.dataset + test_file_name
        edict = {"train": train, "test": test}
        NLP_Dataset = {x: MyDataset(edict[x], tokenizer=tokenizer) for x in ['train', 'test']}
        trainloader = DataLoader(NLP_Dataset['train'], batch_size=opt.batch_size, shuffle=True, drop_last=True)
        testloader = DataLoader(NLP_Dataset['test'], batch_size=opt.batch_size, shuffle=False)
        # print("Index 0 embedding:", model.special_word_embeddings.weight[0].detach().cpu().numpy())
        # print("Index 1 embedding:", model.special_word_embeddings.weight[1].detach().cpu().numpy())

        max_p_emotion, max_r_emotion, max_f1_emotion, max_p_cause, max_r_cause, max_f1_cause, max_p_pair, max_r_pair,\
        max_f1_pair = [-1.] * 9
        optimizer = torch.optim.AdamW(model.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)
        if opt.test_only:
            all_test_logits = torch.tensor([])
            all_test_label = torch.tensor([])
            all_test_mask_label = torch.tensor([])
            all_test_y_bert = torch.tensor([])
            all_test_x_bert = torch.tensor([])
            all_test_cause_gt = torch.tensor([])
            model.eval()
            with torch.no_grad():
                for _, data in enumerate(testloader):
                    x_bert, y_bert, label, mask_label, ECE_x_bert, gt_cause, special_word_ids = data
                    if use_gpu:
                        x_bert = x_bert.cuda()
                        y_bert = y_bert.cuda()
                        label = label.cuda()
                        mask_label = mask_label.cuda()
                        ECE_x_bert = ECE_x_bert.cuda()
                        special_word_ids = special_word_ids.cuda()
                    loss, logits = model(
                        ECE_x_bert,
                        attention_mask=(ECE_x_bert > 0),
                        special_word_ids=special_word_ids,
                        labels=label
                    )
                    logits = F.softmax(logits, dim=-1)
                    all_test_label = torch.cat((all_test_label, label.cpu()), 0)
                    all_test_mask_label = torch.cat((all_test_mask_label, mask_label.cpu()), 0)
                    all_test_logits = torch.cat((all_test_logits, logits.cpu()), 0)
                    all_test_y_bert = torch.cat((all_test_y_bert, y_bert.cpu()), 0)
                    all_test_x_bert = torch.cat((all_test_x_bert, ECE_x_bert.cpu()), 0)
                    all_test_cause_gt = torch.cat((all_test_cause_gt, gt_cause), 0)

                p_cause, r_cause, f_cause = prf_prompt(all_test_logits, all_test_label, all_test_x_bert,
                                                       all_test_cause_gt)
                print("c_p: {:.4f} c_r: {:.4f} c_f: {:.4f}".format(p_cause, r_cause, f_cause))
                if f_cause > max_f1_cause:
                    max_p_cause, max_r_cause, max_f1_cause = p_cause, r_cause, f_cause
                print(
                    "max result---- c_p: {:.4f} c_r: {:.4f} c_f: {:.4f}".format(max_p_cause, max_r_cause, max_f1_cause))
        else:
            for i in range(opt.training_iter):
                model.train()
                start_time, step = time.time(), 1
                for index, data in enumerate(trainloader):
                    with torch.autograd.set_detect_anomaly(True):
                        x_bert, y_bert, label, mask_label, ECE_x_bert, gt_cause, special_word_ids = data
                        if use_gpu:
                            x_bert = x_bert.cuda()
                            y_bert = y_bert.cuda()
                            label = label.cuda()
                            mask_label = mask_label.cuda()
                            ECE_x_bert = ECE_x_bert.cuda()
                            special_word_ids = special_word_ids.cuda()
                        loss, logits = model(
                            ECE_x_bert,
                            attention_mask=(ECE_x_bert > 0),
                            special_word_ids=special_word_ids,
                            labels=mask_label
                        )
                        logits = F.softmax(logits, dim=-1)

                        optimizer.zero_grad()
                        if use_gpu:
                            loss = loss.cuda()
                        loss.backward()
                        optimizer.step()

                        # 减少刷屏: 每20步打印
                        if index % 20 == 0:
                            print("step {}: loss: {:.4f}".format(index, loss))
                            p_cause, r_cause, f_cause = prf_prompt(logits.cpu(), label.cpu(), ECE_x_bert.cpu(),
                                                                   gt_cause)
                            print(
                                "iter: {} c_p: {:.4f} c_r: {:.4f} c_f: {:.4f}".format(index, p_cause, r_cause, f_cause))
                all_test_logits = torch.tensor([])
                all_test_label = torch.tensor([])
                all_test_mask_label = torch.tensor([])
                all_test_y_bert = torch.tensor([])
                all_test_x_bert = torch.tensor([])
                all_test_cause_gt = torch.tensor([])

                model.eval()
                with torch.no_grad():
                    for _, data in enumerate(testloader):
                        x_bert, y_bert, label, mask_label, ECE_x_bert, gt_cause, special_word_ids = data

                        if use_gpu:
                            x_bert = x_bert.cuda()
                            y_bert = y_bert.cuda()
                            label = label.cuda()
                            mask_label = mask_label.cuda()
                            ECE_x_bert = ECE_x_bert.cuda()
                            special_word_ids = special_word_ids.cuda()
                        loss, logits = model(
                            ECE_x_bert,
                            attention_mask=(ECE_x_bert > 0),
                            special_word_ids=special_word_ids,
                            labels=label
                        )

                        # print("Index 0 new:", model.special_word_embeddings.weight[0].detach().cpu().numpy())
                        # print("Index 1 new:", model.special_word_embeddings.weight[1].detach().cpu().numpy())

                        logits = F.softmax(logits, dim=-1)
                        all_test_label = torch.cat((all_test_label, label.cpu()), 0)
                        all_test_mask_label = torch.cat((all_test_mask_label, mask_label.cpu()), 0)
                        all_test_logits = torch.cat((all_test_logits, logits.cpu()), 0)
                        all_test_y_bert = torch.cat((all_test_y_bert, y_bert.cpu()), 0)
                        all_test_x_bert = torch.cat((all_test_x_bert, ECE_x_bert.cpu()), 0)
                        all_test_cause_gt = torch.cat((all_test_cause_gt, gt_cause), 0)

                    p_cause, r_cause, f_cause = prf_prompt(all_test_logits, all_test_label, all_test_x_bert,
                                                           all_test_cause_gt)
                    # print("Index 0 new1:", model.special_word_embeddings.weight[0].detach().cpu().numpy())
                    # print("Index 1 new:1", model.special_word_embeddings.weight[1].detach().cpu().numpy())
                    print("iter{} test result:".format(i))
                    print("c_p: {:.4f} c_r: {:.4f} c_f: {:.4f}".format(p_cause, r_cause, f_cause))

                    if f_cause > max_f1_cause:
                        max_p_cause, max_r_cause, max_f1_cause = p_cause, r_cause, f_cause
                        if opt.savecheckpoint:
                            torch.save(model, save_path + '/' + 'fold{}.pth'.format(fold))
                    print("iter{} test result:".format(i))
                    print(
                        "max result---- c_p: {:.4f} c_r: {:.4f} c_f: {:.4f}".format(max_p_cause, max_r_cause,
                                                                                    max_f1_cause))
        max_result_cause_f.append(max_f1_cause)
        max_result_cause_p.append(max_p_cause)
        max_result_cause_r.append(max_r_cause)

    print("cause")
    print(max_result_cause_f)
    print("average f {:.4f}".format(sum(max_result_cause_f) / len(max_result_cause_f)))
    print(max_result_cause_p)
    print("average p {:.4f}".format(sum(max_result_cause_p) / len(max_result_cause_p)))
    print(max_result_cause_r)
    print("average r {:.4f}".format(sum(max_result_cause_r) / len(max_result_cause_r)))

if __name__ == '__main__':
    run()

