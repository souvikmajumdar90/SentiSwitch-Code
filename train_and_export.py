"""
Trains SentiSwitch (Algorithm 2) on the L3Cube-HingLID dataset with a
proper 80/20 train/test split, then exports everything needed for
standalone inference/evaluation:
  - checkpoints/sentiswitch_gru.pt   (model state_dict)
  - checkpoints/vocab.json           (context/response/lang/sent vocabs)
  - checkpoints/model_config.json    (architecture hyperparameters)
  - data/test_set.json               (held-out test set, for evaluate.py)
"""

import json, re, random
from collections import Counter
from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from functools import partial

random.seed(42)
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def tokenize(text):
    return re.findall(r"\w+|[^\w\s]", text.strip(), re.UNICODE)


def CMI_TARGET_FUNC(context, sentiment):
    return {"pos": 25.0, "neg": 15.0, "neu": 20.0}.get(sentiment.lower(), 20.0)


@dataclass
class Example:
    x: List[str]; y: List[str]; c: str; z: List[str]; s: List[int]; CMI_target: float


def build_dataset_real(D_raw, gold_lid_by_idx):
    D = []
    for idx, (context, response, sentiment) in enumerate(D_raw):
        x, y, c = tokenize(context), tokenize(response), sentiment
        real_tags = gold_lid_by_idx[idx]
        z = [{"HI": "L2", "EN": "L1"}.get(t, "O") for t in real_tags]
        if len(z) != len(y):
            z = (z + ["O"] * len(y))[: len(y)]
        s = [1 if z[t] != z[t + 1] else 0 for t in range(len(y) - 1)]
        s.append(0)
        if len(y) == 0:
            s = []
        D.append(Example(x=x, y=y, c=c, z=z, s=s, CMI_target=CMI_TARGET_FUNC(context, c)))
    return D


class Vocab:
    def __init__(self, specials=["<pad>", "<unk>", "<bos>", "<eos>"]):
        self.itos = list(specials)
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    def build(self, token_lists, min_freq=1):
        counter = Counter(tok for tokens in token_lists for tok in tokens)
        for tok, freq in counter.items():
            if freq >= min_freq and tok not in self.stoi:
                self.stoi[tok] = len(self.itos)
                self.itos.append(tok)
        return self

    def encode(self, tokens, add_bos_eos=False):
        unk = self.stoi.get("<unk>", 0)
        ids = [self.stoi.get(t, unk) for t in tokens]
        if add_bos_eos:
            ids = [self.stoi["<bos>"]] + ids + [self.stoi["<eos>"]]
        return ids

    def __len__(self):
        return len(self.itos)

    def to_dict(self):
        return {"itos": self.itos}

    @classmethod
    def from_dict(cls, d):
        v = cls(specials=[])
        v.itos = d["itos"]
        v.stoi = {t: i for i, t in enumerate(v.itos)}
        return v


class CMDataset(Dataset):
    def __init__(self, examples, cv, rv, lv, sv):
        self.examples, self.cv, self.rv, self.lv, self.sv = examples, cv, rv, lv, sv

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        x_ids = self.cv.encode(ex.x)
        y_ids = self.rv.encode(ex.y, add_bos_eos=True)
        z_ids = self.lv.encode(ex.z)
        s_seq = list(ex.s)
        pad_lang = self.lv.stoi["<pad>"]
        z_ids = [pad_lang] + z_ids + [pad_lang]
        s_seq = [0] + s_seq + [0]
        c_id = self.sv.encode([ex.c])[0]
        return {"x": torch.tensor(x_ids), "y": torch.tensor(y_ids), "z": torch.tensor(z_ids),
                "s": torch.tensor(s_seq, dtype=torch.float), "c": torch.tensor(c_id),
                "cmi_target": torch.tensor(ex.CMI_target, dtype=torch.float),
                "x_len": len(x_ids), "y_len": len(y_ids)}


def make_collate(pad_id_x, pad_id_y, pad_id_z):
    def fn(batch):
        return {"x": pad_sequence([b["x"] for b in batch], batch_first=True, padding_value=pad_id_x),
                "y": pad_sequence([b["y"] for b in batch], batch_first=True, padding_value=pad_id_y),
                "z": pad_sequence([b["z"] for b in batch], batch_first=True, padding_value=pad_id_z),
                "s": pad_sequence([b["s"] for b in batch], batch_first=True, padding_value=0.0),
                "c": torch.stack([b["c"] for b in batch]),
                "cmi_target": torch.stack([b["cmi_target"] for b in batch]),
                "x_len": torch.tensor([b["x_len"] for b in batch]),
                "y_len": torch.tensor([b["y_len"] for b in batch])}
    return fn


class Encoder(nn.Module):
    def __init__(self, vsize, emb_dim, hid, pad_id):
        super().__init__()
        self.embedding = nn.Embedding(vsize, emb_dim, padding_idx=pad_id)
        self.rnn = nn.GRU(emb_dim, hid, batch_first=True, bidirectional=True)

    def forward(self, x, x_len):
        emb = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(emb, x_len.cpu(), batch_first=True, enforce_sorted=False)
        outputs, h_n = self.rnn(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        return outputs, torch.cat([h_n[0], h_n[1]], dim=-1)


class Attention(nn.Module):
    def __init__(self, hid):
        super().__init__()
        self.attn = nn.Linear(hid * 3, hid)
        self.v = nn.Linear(hid, 1, bias=False)

    def forward(self, dec_hidden, enc_outputs, enc_mask):
        T_x = enc_outputs.size(1)
        rep = dec_hidden.unsqueeze(1).repeat(1, T_x, 1)
        energy = torch.tanh(self.attn(torch.cat([rep, enc_outputs], dim=-1)))
        scores = self.v(energy).squeeze(-1).masked_fill(enc_mask == 0, -1e9)
        w = F.softmax(scores, dim=-1)
        return torch.bmm(w.unsqueeze(1), enc_outputs).squeeze(1), w


class Decoder(nn.Module):
    def __init__(self, vsize, emb_dim, hid, n_lang, sent_emb_dim, pad_id):
        super().__init__()
        self.embedding = nn.Embedding(vsize, emb_dim, padding_idx=pad_id)
        self.attention = Attention(hid)
        self.rnn = nn.GRUCell(emb_dim + hid * 2 + sent_emb_dim, hid)
        self.gen_head = nn.Linear(hid * 3, vsize)
        self.lid_head = nn.Linear(hid, n_lang)
        self.switch_head = nn.Linear(hid, 1)

    def forward_step(self, y_t, h_prev, enc_outputs, enc_mask, sent_emb):
        emb = self.embedding(y_t)
        context, w = self.attention(h_prev, enc_outputs, enc_mask)
        h_t = self.rnn(torch.cat([emb, context, sent_emb], dim=-1), h_prev)
        return h_t, self.gen_head(torch.cat([h_t, context], dim=-1)), self.lid_head(h_t), self.switch_head(h_t), w


class SwitchingAwareSCNLG(nn.Module):
    def __init__(self, cvs, rvs, n_lang, n_sent, emb_dim=96, hid=160, sent_emb_dim=16, pad_id_x=0, pad_id_y=0):
        super().__init__()
        self.encoder = Encoder(cvs, emb_dim, hid, pad_id_x)
        self.decoder = Decoder(rvs, emb_dim, hid, n_lang, sent_emb_dim, pad_id_y)
        self.sent_classifier = nn.Linear(hid * 2, n_sent)
        self.sent_embedding = nn.Embedding(n_sent, sent_emb_dim)
        self.bridge = nn.Linear(hid * 2, hid)

    def forward(self, x, x_len, y, c):
        enc_mask = (x != 0).float()
        enc_outputs, h_final = self.encoder(x, x_len)
        c_logits = self.sent_classifier(h_final)
        sent_emb = self.sent_embedding(c)
        h_t = torch.tanh(self.bridge(h_final))
        y_l, z_l, s_l = [], [], []
        for t in range(y.size(1) - 1):
            h_t, gl, ll, sl, _ = self.decoder.forward_step(y[:, t], h_t, enc_outputs, enc_mask, sent_emb)
            y_l.append(gl.unsqueeze(1)); z_l.append(ll.unsqueeze(1)); s_l.append(sl.unsqueeze(1))
        return h_final, torch.cat(y_l, 1), torch.cat(z_l, 1), torch.cat(s_l, 1), c_logits


def compute_cmi_batch(z_ids, lang_vocab, pad_id_z):
    B = z_ids.size(0); dev = z_ids.device; o_id = lang_vocab.stoi.get("O", -1)
    out = torch.zeros(B, device=dev)
    for b in range(B):
        v = z_ids[b][z_ids[b] != pad_id_z]
        if o_id != -1:
            v = v[v != o_id]
        if v.numel() == 0:
            continue
        counts = torch.bincount(v)
        out[b] = 100.0 * (1.0 - counts.max().float() / v.numel())
    return out


# =====================================================================
# Load dataset (reuses the same L3Cube-HingLID-derived D_raw + gold LID
# tags and the same 80/20 split used throughout this conversation)
# =====================================================================
with open("data/hinglid_D_raw.json", "r", encoding="utf-8") as f:
    payload = json.load(f)

D_raw_all = [tuple(x) for x in payload["D_raw"]]
gold_lid_all = {int(k): v for k, v in payload["gold_lid_by_response_idx"].items()}

indices = list(range(len(D_raw_all)))
random.shuffle(indices)
split_point = int(0.8 * len(indices))
train_idx, test_idx = indices[:split_point], indices[split_point:]

D_raw_train = [D_raw_all[i] for i in train_idx]
D_raw_test = [D_raw_all[i] for i in test_idx]
gold_lid_train = {new_i: gold_lid_all[old_i] for new_i, old_i in enumerate(train_idx)}

print(f"Train: {len(D_raw_train)}  Test: {len(D_raw_test)}")

D_train = build_dataset_real(D_raw_train, gold_lid_train)

context_vocab = Vocab().build([ex.x for ex in D_train])
response_vocab = Vocab().build([ex.y for ex in D_train])
lang_vocab = Vocab(specials=["<pad>"]).build([ex.z for ex in D_train])
sent_vocab = Vocab(specials=["<pad>"]).build([[ex.c] for ex in D_train])
pad_id_x = context_vocab.stoi["<pad>"]
pad_id_y = response_vocab.stoi["<pad>"]
pad_id_z = lang_vocab.stoi["<pad>"]

collate_fn = make_collate(pad_id_x, pad_id_y, pad_id_z)
dataloader = DataLoader(CMDataset(D_train, context_vocab, response_vocab, lang_vocab, sent_vocab),
                         batch_size=8, shuffle=True, collate_fn=collate_fn)

EMB_DIM, HID, SENT_EMB_DIM = 96, 160, 16
model = SwitchingAwareSCNLG(len(context_vocab), len(response_vocab), len(lang_vocab), len(sent_vocab),
                             emb_dim=EMB_DIM, hid=HID, sent_emb_dim=SENT_EMB_DIM,
                             pad_id_x=pad_id_x, pad_id_y=pad_id_y).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
NUM_EPOCHS = 40

for epoch in range(1, NUM_EPOCHS + 1):
    for batch in dataloader:
        x, y, z, s, c = (batch[k].to(device) for k in ["x", "y", "z", "s", "c"])
        cmi_target, x_len = batch["cmi_target"].to(device), batch["x_len"].to(device)
        _, y_logits, z_logits, s_logits, c_logits = model(x, x_len, y, c)
        y_t, z_t, s_t = y[:, 1:], z[:, 1:], s[:, 1:]
        L_gen = F.cross_entropy(y_logits.reshape(-1, y_logits.size(-1)), y_t.reshape(-1), ignore_index=pad_id_y)
        L_sent = F.cross_entropy(c_logits, c)
        L_lid = F.cross_entropy(z_logits.reshape(-1, z_logits.size(-1)), z_t.reshape(-1), ignore_index=pad_id_z)
        mask = (y_t != pad_id_y).float()
        L_switch = (F.binary_cross_entropy_with_logits(s_logits.squeeze(-1), s_t, reduction="none") * mask).sum() / mask.sum().clamp(min=1.0)
        L_cm = F.mse_loss(compute_cmi_batch(z_t, lang_vocab, pad_id_z), cmi_target)
        loss = L_gen + 0.5 * L_sent + 1.0 * L_lid + 1.0 * L_switch + 0.1 * L_cm
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
    if epoch % 10 == 0 or epoch == NUM_EPOCHS:
        print(f"Epoch {epoch}/{NUM_EPOCHS} | loss={loss.item():.4f}")

# =====================================================================
# Export everything for standalone inference.py / evaluate.py
# =====================================================================
torch.save(model.state_dict(), "checkpoints/sentiswitch_gru.pt")

with open("checkpoints/vocab.json", "w", encoding="utf-8") as f:
    json.dump({
        "context_vocab": context_vocab.to_dict(),
        "response_vocab": response_vocab.to_dict(),
        "lang_vocab": lang_vocab.to_dict(),
        "sent_vocab": sent_vocab.to_dict(),
    }, f, ensure_ascii=False, indent=2)

with open("checkpoints/model_config.json", "w") as f:
    json.dump({
        "emb_dim": EMB_DIM, "hid": HID, "sent_emb_dim": SENT_EMB_DIM,
        "context_vocab_size": len(context_vocab), "response_vocab_size": len(response_vocab),
        "n_lang": len(lang_vocab), "n_sent": len(sent_vocab),
        "pad_id_x": pad_id_x, "pad_id_y": pad_id_y, "pad_id_z": pad_id_z,
        "architecture": "custom GRU encoder-decoder with attention (NOT mT5 -- "
                         "see README for why this repo layout names the checkpoint "
                         "mt5_sentiswitch.pt but the actual model is a lightweight "
                         "GRU seq2seq)",
        "trained_on": "L3Cube-HingLID (real gold LID tags, heuristic sentiment, "
                       "synthetic consecutive-sentence context/response pairing)",
        "n_train_examples": len(D_raw_train),
    }, f, indent=2)

with open("data/test_set.json", "w", encoding="utf-8") as f:
    json.dump(D_raw_test, f, ensure_ascii=False, indent=2)

print(f"\nExported checkpoint, vocab, config to checkpoints/")
print(f"Exported {len(D_raw_test)}-example held-out test set to data/test_set.json")
