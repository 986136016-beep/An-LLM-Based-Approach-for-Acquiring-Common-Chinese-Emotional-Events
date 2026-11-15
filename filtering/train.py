
import os
import csv
import argparse
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, precision_score, recall_score

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          get_linear_schedule_with_warmup)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=str, default='./robert')
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--lr', type=float, default=4e-5)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--max_len', type=int, default=64)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--patience', type=int, default=2)
    p.add_argument('--data_dir', type=str, default='.')
    p.add_argument('--prefix', type=str, default='D3',
                   help='数据文件前缀: D3(混合) / D3_nonneutral / D3_bei')
    p.add_argument('--device', type=str, default='cuda')
    return p.parse_args()

class ECDataset(Dataset):
    def __init__(self, path, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.texts, self.labels = [], []
        with open(path, encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for r in reader:
                self.texts.append(r['event'])
                self.labels.append(int(r['label']))
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, i):
        enc = self.tokenizer(self.texts[i], max_length=self.max_len,
                             padding='max_length', truncation=True,
                             return_tensors='pt')
        return {
            'input_ids': enc['input_ids'][0],
            'attention_mask': enc['attention_mask'][0],
            'label': torch.tensor(self.labels[i], dtype=torch.long),
        }

def evaluate(model, loader, device):
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            logits = model(input_ids, attention_mask=mask).logits
            p = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            probs.extend(p.tolist())
            labels.extend(batch['label'].cpu().numpy().tolist())
    probs = np.array(probs)
    labels = np.array(labels)
    ap = average_precision_score(labels, probs)

    pred = (probs >= 0.5).astype(int)
    prec = precision_score(labels, pred, zero_division=0)
    rec = recall_score(labels, pred, zero_division=0)
    return ap, prec, rec, probs, labels

def find_best_threshold(probs, labels):

    precision, recall, thresholds = precision_recall_curve(labels, probs)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    idx = np.argmax(f1)
    best_thr = thresholds[idx] if idx < len(thresholds) else 0.5
    return best_thr

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print('device:', device)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)
    model.config.hidden_dropout_prob = args.dropout
    model.config.attention_probs_dropout_prob = args.dropout
    model.to(device)

    train_ds = ECDataset(os.path.join(args.data_dir, args.prefix + '_train.csv'), tokenizer, args.max_len)
    val_ds = ECDataset(os.path.join(args.data_dir, args.prefix + '_val.csv'), tokenizer, args.max_len)
    test_ds = ECDataset(os.path.join(args.data_dir, args.prefix + '_test.csv'), tokenizer, args.max_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps*0.1), num_training_steps=total_steps)

    best_ap = -1
    best_threshold = 0.5
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids, attention_mask=mask, labels=labels)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        ap, prec, rec, val_probs, val_labels = evaluate(model, val_loader, device)
        print('epoch %d: loss=%.4f  val AP=%.4f  P=%.4f  R=%.4f' % (epoch, total_loss/len(train_loader), ap, prec, rec))

        if ap > best_ap:
            best_ap = ap
            best_threshold = find_best_threshold(val_probs, val_labels)
            torch.save(model.state_dict(), args.prefix + '_best_model.pt')

            np.save(args.prefix + '_val_probs.npy', val_probs)
            np.save(args.prefix + '_val_labels.npy', val_labels)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print('early stopping at epoch %d' % epoch)
                break

    model.load_state_dict(torch.load(args.prefix + '_best_model.pt'))
    ap, prec, rec, test_probs, test_labels = evaluate(model, test_loader, device)

    from sklearn.metrics import precision_score, recall_score, f1_score
    test_pred = (test_probs >= best_threshold).astype(int)
    test_p = precision_score(test_labels, test_pred, zero_division=0)
    test_r = recall_score(test_labels, test_pred, zero_division=0)
    test_f1 = f1_score(test_labels, test_pred, zero_division=0)

    np.save(args.prefix + '_test_probs.npy', test_probs)
    np.save(args.prefix + '_test_labels.npy', test_labels)

    print()
    print('=' * 50)
    print('[%s] 最佳threshold(validation set上): %.4f' % (args.prefix, best_threshold))
    print('[%s] test set: PR-AUC=%.4f  P(threshold)=%.4f  R(threshold)=%.4f  F1(threshold)=%.4f' % (
        args.prefix, ap, test_p, test_r, test_f1))
    print('=' * 50)

    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    lines = []
    lines.append('prefix: %s' % args.prefix)
    lines.append('best_threshold(val): %.4f' % best_threshold)
    lines.append('test PR-AUC: %.4f' % ap)
    lines.append('test Precision(best_thr): %.4f' % test_p)
    lines.append('test Recall(best_thr): %.4f' % test_r)
    lines.append('test F1(best_thr): %.4f' % test_f1)
    lines.append('')
    lines.append('不同threshold下test set P/R/F1:')
    lines.append('threshold\tprecision\trecall\tf1')
    for thr in thresholds:
        pred = (test_probs >= thr).astype(int)
        p = precision_score(test_labels, pred, zero_division=0)
        r = recall_score(test_labels, pred, zero_division=0)
        f = f1_score(test_labels, pred, zero_division=0)
        lines.append('%.2f\t%.4f\t%.4f\t%.4f' % (thr, p, r, f))
        print('  threshold %.2f: P=%.4f  R=%.4f  F1=%.4f' % (thr, p, r, f))

    with open(args.prefix + '_result.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print('result已written %s_result.txt (含多threshold对比)' % args.prefix)

if __name__ == '__main__':
    main()
