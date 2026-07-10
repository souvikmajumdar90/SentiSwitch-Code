"""
inference.py -- Standalone Algorithm 3 (Switching-Aware Sentiment-
Controlled Beam Search) inference using the trained checkpoint.

Usage:
    python3 inference.py --input sample_test_inputs.json --output sample_outputs.json

Input JSON format (list of objects):
    [{"context": "...", "sentiment": "pos"}, ...]
    sentiment must be one of: pos, neg, neu

Output JSON format:
    [{"context": "...", "sentiment": "pos", "generated": "...",
      "lang_tags": [...], "cmi_achieved": 23.5, "score": -4.12}, ...]
"""

import argparse
import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def tokenize(text):
    return re.findall(r"\w+|[^\w\s]", text.strip(), re.UNICODE)


# =====================================================================
# Model architecture (must match train_and_export.py exactly)
# =====================================================================
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


class Vocab:
    def __init__(self):
        self.itos = []
        self.stoi = {}

    @classmethod
    def from_dict(cls, d):
        v = cls()
        v.itos = d["itos"]
        v.stoi = {t: i for i, t in enumerate(v.itos)}
        return v

    def encode(self, tokens, add_bos_eos=False):
        unk = self.stoi.get("<unk>", 0)
        ids = [self.stoi.get(t, unk) for t in tokens]
        if add_bos_eos:
            ids = [self.stoi["<bos>"]] + ids + [self.stoi["<eos>"]]
        return ids

    def __len__(self):
        return len(self.itos)


def load_model(checkpoint_dir="checkpoints"):
    with open(f"{checkpoint_dir}/model_config.json") as f:
        cfg = json.load(f)
    with open(f"{checkpoint_dir}/vocab.json") as f:
        vocabs = json.load(f)

    context_vocab = Vocab.from_dict(vocabs["context_vocab"])
    response_vocab = Vocab.from_dict(vocabs["response_vocab"])
    lang_vocab = Vocab.from_dict(vocabs["lang_vocab"])
    sent_vocab = Vocab.from_dict(vocabs["sent_vocab"])

    model = SwitchingAwareSCNLG(
        cfg["context_vocab_size"], cfg["response_vocab_size"], cfg["n_lang"], cfg["n_sent"],
        emb_dim=cfg["emb_dim"], hid=cfg["hid"], sent_emb_dim=cfg["sent_emb_dim"],
        pad_id_x=cfg["pad_id_x"], pad_id_y=cfg["pad_id_y"],
    ).to(device)
    model.load_state_dict(torch.load(f"{checkpoint_dir}/sentiswitch_gru.pt", map_location=device))
    model.eval()

    return model, context_vocab, response_vocab, lang_vocab, sent_vocab, cfg


# =====================================================================
# Algorithm 3: Switching-Aware Sentiment-Controlled Beam Search
# =====================================================================
def compute_cmi_from_ids(z_ids, o_id):
    valid = [z for z in z_ids if z != o_id]
    if not valid:
        return 0.0
    counts = Counter(valid)
    return 100.0 * (1 - max(counts.values()) / len(valid))


def switch_consistency(switch_probs, s_gold, eps=1e-8):
    if not switch_probs:
        return 0.0
    n = min(len(switch_probs), len(s_gold))
    total = 0.0
    for p, s in zip(switch_probs[:n], s_gold[:n]):
        p = min(max(p, eps), 1 - eps)
        total += s * math.log(p) + (1 - s) * math.log(1 - p)
    return total / n


@dataclass
class BeamHypothesis:
    y: List[int]; score: float; z: List[int]; s: List[int]
    h_state: torch.Tensor; finished: bool = False
    logP_gen_sum: float = 0.0; switch_probs: List[float] = field(default_factory=list)


def switching_aware_sentiment_beam_search(model, context_str, target_sentiment,
        context_vocab, response_vocab, lang_vocab, sent_vocab,
        K=4, T_max=20, alpha=1.0, beta=0.05, gamma=1.0, cmi_target=20.0, M=None, eps=1e-8):
    model.eval()
    M = M if M is not None else max(2 * K, K)
    bos_id, eos_id = response_vocab.stoi["<bos>"], response_vocab.stoi["<eos>"]
    o_id = lang_vocab.stoi.get("O", -1)
    c_id = sent_vocab.stoi.get(target_sentiment, sent_vocab.stoi.get("<pad>", 0))
    with torch.no_grad():
        x_ids = context_vocab.encode(tokenize(context_str))
        x_t = torch.tensor([x_ids], dtype=torch.long, device=device)
        x_l = torch.tensor([len(x_ids)], dtype=torch.long, device=device)
        enc_mask = (x_t != 0).float()
        enc_outputs, h_final = model.encoder(x_t, x_l)
        c_tensor = torch.tensor([c_id], dtype=torch.long, device=device)
        sent_emb = model.sent_embedding(c_tensor)
        c_logits = model.sent_classifier(h_final)
        p_sent = F.softmax(c_logits, dim=-1).squeeze(0)[c_id].item()
        h0 = torch.tanh(model.bridge(h_final))
        B = [BeamHypothesis(y=[bos_id], score=0.0, z=[], s=[], h_state=h0)]
        for t in range(1, T_max + 1):
            NewB = []
            for h in B:
                if h.finished:
                    NewB.append(h)
                    continue
                y_t = torch.tensor([h.y[-1]], dtype=torch.long, device=device)
                h_t, gl, ll, sl, _ = model.decoder.forward_step(y_t, h.h_state, enc_outputs, enc_mask, sent_emb)
                log_probs = F.log_softmax(gl, dim=-1).squeeze(0)
                lid_lp = F.log_softmax(ll, dim=-1).squeeze(0)
                switch_prob = torch.sigmoid(sl).item()
                topv, topi = log_probs.topk(min(M, log_probs.size(-1)))
                z_pred = lid_lp.argmax().item()
                for logp, tok_id in zip(topv.tolist(), topi.tolist()):
                    y_new = h.y + [tok_id]
                    z_new = h.z + [z_pred]
                    s_new = [1 if z_new[j] != z_new[j + 1] else 0 for j in range(len(z_new) - 1)]
                    s_new.append(0)
                    logP_gen = h.logP_gen_sum + logp
                    cmi_curr = compute_cmi_from_ids(z_new, o_id)
                    switch_probs_new = h.switch_probs + [switch_prob]
                    phi = switch_consistency(switch_probs_new, s_new, eps)
                    J = logP_gen + alpha * math.log(p_sent + eps) - beta * abs(cmi_curr - cmi_target) + gamma * phi
                    finished = (tok_id == eos_id)
                    NewB.append(BeamHypothesis(y=y_new, score=J, z=z_new, s=s_new, h_state=h_t,
                                                finished=finished, logP_gen_sum=logP_gen,
                                                switch_probs=switch_probs_new))
            NewB.sort(key=lambda hh: hh.score, reverse=True)
            B = NewB[:K]
            if all(hh.finished for hh in B):
                break
        h_best = max(B, key=lambda hh: hh.score)
        token_ids = [tid for tid in h_best.y if tid not in (bos_id, eos_id)]
        tokens = [response_vocab.itos[tid] for tid in token_ids]
        lang_tags = [lang_vocab.itos[z] for z in h_best.z[: len(tokens)]]
        return {
            "tokens": tokens, "text": " ".join(tokens), "lang_tags": lang_tags,
            "cmi_achieved": compute_cmi_from_ids(h_best.z[: len(tokens)], o_id),
            "score": h_best.score,
        }


def main():
    parser = argparse.ArgumentParser(description="SentiSwitch standalone inference")
    parser.add_argument("--input", default="sample_test_inputs.json",
                         help="JSON file: list of {'context': str, 'sentiment': 'pos'|'neg'|'neu'}")
    parser.add_argument("--output", default="sample_outputs.json")
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--beam_size", type=int, default=4)
    parser.add_argument("--max_len", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=1.0, help="sentiment weight")
    parser.add_argument("--beta", type=float, default=0.05, help="CMI weight")
    parser.add_argument("--gamma", type=float, default=1.0, help="switch-consistency weight")
    parser.add_argument("--cmi_target", type=float, default=20.0)
    args = parser.parse_args()

    model, context_vocab, response_vocab, lang_vocab, sent_vocab, cfg = load_model(args.checkpoint_dir)
    print(f"Loaded model trained on: {cfg['trained_on']}")
    print(f"Architecture: {cfg['architecture']}\n")

    with open(args.input, "r", encoding="utf-8") as f:
        inputs = json.load(f)

    outputs = []
    for item in inputs:
        context, sentiment = item["context"], item["sentiment"]
        result = switching_aware_sentiment_beam_search(
            model, context, sentiment, context_vocab, response_vocab, lang_vocab, sent_vocab,
            K=args.beam_size, T_max=args.max_len, alpha=args.alpha, beta=args.beta,
            gamma=args.gamma, cmi_target=args.cmi_target,
        )
        print(f"Context   : {context}")
        print(f"Sentiment : {sentiment}")
        print(f"Generated : {result['text']}")
        print(f"CMI       : {result['cmi_achieved']:.2f}\n")
        outputs.append({
            "context": context, "sentiment": sentiment, "generated": result["text"],
            "lang_tags": result["lang_tags"], "cmi_achieved": result["cmi_achieved"],
            "score": result["score"],
        })

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(outputs, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(outputs)} generations to {args.output}")


if __name__ == "__main__":
    main()
