import argparse
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

from src.utils.logging_utils import get_logger

log = get_logger(__name__)
LABELS = ['refusal', 'partial_compliance', 'full_violation']


class JudgeDataset(Dataset):
    def __init__(self, path, tokenizer, max_len=512):
        self.rows = [json.loads(l) for l in open(path)]
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        text = f"PROMPT: {row['prompt']}\nRESPONSE: {row['response']}"
        enc = self.tokenizer(text, truncation=True, max_length=self.max_len, padding='max_length', return_tensors='pt')
        label = LABELS.index(row['label'])
        return {'input_ids': enc['input_ids'].squeeze(0), 'attention_mask': enc['attention_mask'].squeeze(0), 'label': torch.tensor(label, dtype=torch.long)}


class JudgeModel(nn.Module):
    def __init__(self, base_model_name='distilbert-base-uncased', n_labels=3):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model_name)
        hidden = self.encoder.config.hidden_size
        self.classifier = nn.Linear(hidden, n_labels)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return self.classifier(cls)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = (0, 0)
    for batch in loader:
        logits = model(batch['input_ids'].to(device), batch['attention_mask'].to(device))
        preds = logits.argmax(dim=-1).cpu()
        correct += (preds == batch['label']).sum().item()
        total += len(preds)
    return correct / max(1, total)


def main(train_path, val_path, out_dir, epochs=3, batch_size=16, lr=2e-05, max_len=512, seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    model = JudgeModel().to(device)
    train_ds = JudgeDataset(train_path, tokenizer, max_len=max_len)
    val_ds = JudgeDataset(val_path, tokenizer, max_len=max_len)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            logits = model(batch['input_ids'].to(device), batch['attention_mask'].to(device))
            loss = loss_fn(logits, batch['label'].to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        acc = evaluate(model, val_loader, device)
        log.info('epoch %d: train_loss=%.4f val_acc=%.4f', epoch, total_loss / len(train_loader), acc)
    import os
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), f'{out_dir}/judge_latest.pt')
    log.info('reward judge saved to %s', out_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default='data/judge_train/train.jsonl')
    parser.add_argument('--val', default='data/judge_train/val.jsonl')
    parser.add_argument('--out_dir', default='checkpoints/judge_model')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=2e-05)
    parser.add_argument('--max_len', type=int, default=512)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    main(args.train, args.val, args.out_dir, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, max_len=args.max_len, seed=args.seed)
