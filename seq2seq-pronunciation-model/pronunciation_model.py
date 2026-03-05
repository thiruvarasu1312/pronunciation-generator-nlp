# ================================================================
# 1. Setup
# ================================================================


import nltk
nltk.download('cmudict')
from nltk.corpus import cmudict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import random

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# ================================================================
# 2. Load CMU Pronouncing Dictionary
# ================================================================
cmu = cmudict.dict()
print("Total words in CMU dict:", len(cmu))

# Example entry
print("cat ->", cmu["cat"])

# Prepare parallel pairs: spelling (chars) → phonemes (flattened)
pairs = []
for word, pron in cmu.items():
    # Use first pronunciation only
    spelling = list(word)   # character-level
    phonemes = pron[0]      # list of phoneme tokens
    phonemes = " ".join(phonemes)  # turn into string
    pairs.append(("".join(spelling), phonemes))

print("Example pair:", pairs[0])

# For low-resource condition, we can SUBSAMPLE
pairs = pairs[:5000]   # take only 5k pairs
print("Using", len(pairs), "pairs for training")

# ================================================================
# 3. Vocabulary
# ================================================================
SOS = "<sos>"
EOS = "<eos>"
PAD = "<pad>"

def build_vocab(data):
    chars = set()
    for src, tgt in data:
        chars.update(list(src))
        chars.update(list(tgt))
    vocab = [PAD, SOS, EOS] + sorted(list(chars))
    stoi = {ch:i for i,ch in enumerate(vocab)}
    itos = {i:ch for i,ch in enumerate(vocab)}
    return vocab, stoi, itos

vocab, stoi, itos = build_vocab(pairs)
VOCAB_SIZE = len(vocab)
print("Vocab size:", VOCAB_SIZE)

def encode(seq):
    return [stoi[c] for c in seq]

def tensorize(seq):
    return torch.tensor(seq, dtype=torch.long)

# ================================================================
# 4. Dataset & DataLoader
# ================================================================
class CharDataset(Dataset):
    def __init__(self, pairs):
        self.data = pairs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src, tgt = self.data[idx]
        src_ids = encode(src) + [stoi[EOS]]
        tgt_ids = [stoi[SOS]] + encode(tgt) + [stoi[EOS]]
        return tensorize(src_ids), tensorize(tgt_ids)

def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_batch = pad_sequence(src_batch, padding_value=stoi[PAD], batch_first=True)
    tgt_batch = pad_sequence(tgt_batch, padding_value=stoi[PAD], batch_first=True)
    src_lens = torch.tensor([len(x) for x in src_batch])
    return src_batch, src_lens, tgt_batch

train_loader = DataLoader(CharDataset(pairs), batch_size=32, shuffle=True, collate_fn=collate_fn)

# ================================================================
# 5. Seq2Seq with Attention
# ================================================================
class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_size, hid):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_size, padding_idx=stoi[PAD])
        self.rnn = nn.GRU(emb_size, hid, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hid*2, hid)

    def forward(self, src, lengths):
        embedded = self.emb(src)
        enc_out, hidden = self.rnn(embedded)
        h_cat = torch.cat((hidden[-2], hidden[-1]), dim=1)
        dec_init = torch.tanh(self.fc(h_cat)).unsqueeze(0)
        return enc_out, dec_init

class Attention(nn.Module):
    def __init__(self, enc_hid, dec_hid):
        super().__init__()
        self.W1 = nn.Linear(enc_hid, dec_hid)
        self.W2 = nn.Linear(dec_hid, dec_hid)
        self.v = nn.Linear(dec_hid, 1, bias=False)

    def forward(self, enc_out, hidden):
        dec_h = hidden.squeeze(0).unsqueeze(1)
        score = self.v(torch.tanh(self.W1(enc_out) + self.W2(dec_h)))
        attn = F.softmax(score, dim=1)
        context = (attn * enc_out).sum(dim=1)
        return context, attn

class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_size, enc_hid, dec_hid):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_size, padding_idx=stoi[PAD])
        self.rnn = nn.GRU(emb_size+enc_hid, dec_hid, batch_first=True)
        self.fc = nn.Linear(emb_size+enc_hid+dec_hid, vocab_size)
        self.attn = Attention(enc_hid, dec_hid)

    def forward(self, input_tok, hidden, enc_out):
        emb = self.emb(input_tok).unsqueeze(1)
        context, _ = self.attn(enc_out, hidden)
        rnn_input = torch.cat([emb, context.unsqueeze(1)], dim=2)
        output, hidden = self.rnn(rnn_input, hidden)
        output = output.squeeze(1)
        logits = self.fc(torch.cat([output, context, emb.squeeze(1)], dim=1))
        return logits, hidden

class Seq2Seq(nn.Module):
    def __init__(self, enc, dec):
        super().__init__()
        self.enc = enc
        self.dec = dec

    def forward(self, src, src_lens, tgt, teacher_forcing=0.8):
        B, Tt = tgt.size()
        outputs = torch.zeros(B, Tt, VOCAB_SIZE).to(DEVICE)
        enc_out, hidden = self.enc(src, src_lens)
        input_tok = tgt[:,0]
        for t in range(1, Tt):
            logits, hidden = self.dec(input_tok, hidden, enc_out)
            outputs[:,t,:] = logits
            teacher = random.random() < teacher_forcing
            top1 = logits.argmax(1)
            input_tok = tgt[:,t] if teacher else top1
        return outputs

# ================================================================
# 6. Training
# ================================================================
EMB, HID = 64, 128
enc = Encoder(VOCAB_SIZE, EMB, HID).to(DEVICE)
dec = Decoder(VOCAB_SIZE, EMB, HID*2, HID).to(DEVICE)
model = Seq2Seq(enc, dec).to(DEVICE)

criterion = nn.CrossEntropyLoss(ignore_index=stoi[PAD])
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def train(model, loader, epochs=10):
    for ep in range(1, epochs+1):
        total_loss = 0
        for src, src_lens, tgt in loader:
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            outputs = model(src, src_lens, tgt)
            loss = criterion(outputs.view(-1, VOCAB_SIZE), tgt.view(-1))
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {ep}, Loss {total_loss/len(loader):.4f}")

train(model, train_loader, epochs=10)

# ================================================================
# 7. Inference
# ================================================================
def translate(word, max_len=30):
    model.eval()
    src = tensorize(encode(word)+[stoi[EOS]]).unsqueeze(0).to(DEVICE)
    src_len = torch.tensor([src.size(1)]).to(DEVICE)
    enc_out, hidden = model.enc(src, src_len)
    input_tok = torch.tensor([stoi[SOS]], device=DEVICE)
    result = []
    for _ in range(max_len):
        logits, hidden = model.dec(input_tok, hidden, enc_out)
        top1 = logits.argmax(1)
        char = itos[top1.item()]
        if char == EOS: break
        result.append(char)
        input_tok = top1
    return "".join(result)

print("cat ->", translate("cat"))
print("dog ->", translate("dog"))
print("fish ->", translate("fish"))
