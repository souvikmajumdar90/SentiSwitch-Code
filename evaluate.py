"""
evaluate.py -- Computes MIPE / BLEU / ROUGE-1/2/L for the trained
SentiSwitch checkpoint against a held-out test set.

Usage:
    python3 evaluate.py --test_file data/test_set.json

test_set.json format: list of [context, gold_response, sentiment] triples
(this is exactly what train_and_export.py exports for the held-out split).

MIPE is a custom metric defined here as:
    MIPE = 1 - |1 - p(target_sentiment)|
using the model's own sentiment-classifier probability for the target
class. It is NOT a standardized published metric -- see README.md.
"""

import argparse
import json

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

from inference import load_model, switching_aware_sentiment_beam_search, tokenize


def mipe_score(model, context_str, sentiment, context_vocab, sent_vocab, device):
    import torch
    import torch.nn.functional as F
    x_ids = context_vocab.encode(tokenize(context_str))
    with torch.no_grad():
        x_t = torch.tensor([x_ids], dtype=torch.long, device=device)
        x_l = torch.tensor([len(x_ids)], dtype=torch.long, device=device)
        _, h_final = model.encoder(x_t, x_l)
        probs = F.softmax(model.sent_classifier(h_final), dim=-1).squeeze(0)
        p = probs[sent_vocab.stoi.get(sentiment, 0)].item()
    return 1.0 - abs(1.0 - p)


def main():
    parser = argparse.ArgumentParser(description="Evaluate SentiSwitch on a held-out test set")
    parser.add_argument("--test_file", default="data/test_set.json")
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--beam_size", type=int, default=4)
    parser.add_argument("--max_len", type=int, default=20)
    parser.add_argument("--output", default="evaluation_results.json")
    args = parser.parse_args()

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, context_vocab, response_vocab, lang_vocab, sent_vocab, cfg = load_model(args.checkpoint_dir)
    print(f"Loaded model trained on: {cfg['trained_on']} ({cfg['n_train_examples']} examples)")
    print(f"Architecture: {cfg['architecture']}\n")

    with open(args.test_file, "r", encoding="utf-8") as f:
        test_set = json.load(f)
    print(f"Evaluating on {len(test_set)} held-out examples\n")

    smoothie = SmoothingFunction().method4
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    bleu_scores, r1_scores, r2_scores, rl_scores, mipe_scores = [], [], [], [], []
    per_example = []

    for context, gold_response, sentiment in test_set:
        result = switching_aware_sentiment_beam_search(
            model, context, sentiment, context_vocab, response_vocab, lang_vocab, sent_vocab,
            K=args.beam_size, T_max=args.max_len,
        )
        hyp_tokens = result["tokens"] or ["<empty>"]
        ref_tokens = tokenize(gold_response)

        bleu = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothie)
        rouge = scorer.score(" ".join(ref_tokens), " ".join(hyp_tokens))
        mipe = mipe_score(model, context, sentiment, context_vocab, sent_vocab, device)

        bleu_scores.append(bleu)
        r1_scores.append(rouge["rouge1"].fmeasure)
        r2_scores.append(rouge["rouge2"].fmeasure)
        rl_scores.append(rouge["rougeL"].fmeasure)
        mipe_scores.append(mipe)

        per_example.append({
            "context": context, "gold": gold_response, "generated": result["text"],
            "sentiment": sentiment, "bleu": bleu, "rouge1": rouge["rouge1"].fmeasure,
            "rouge2": rouge["rouge2"].fmeasure, "rougeL": rouge["rougeL"].fmeasure, "mipe": mipe,
        })

    avg = lambda lst: sum(lst) / len(lst) if lst else 0.0
    summary = {
        "n_examples": len(test_set),
        "MIPE": avg(mipe_scores),
        "BLEU": avg(bleu_scores),
        "ROUGE-1": avg(r1_scores),
        "ROUGE-2": avg(r2_scores),
        "ROUGE-L": avg(rl_scores),
    }

    print("=" * 50)
    print("EVALUATION SUMMARY (held-out test set)")
    print("=" * 50)
    for k, v in summary.items():
        print(f"{k:<12}: {v:.4f}" if isinstance(v, float) else f"{k:<12}: {v}")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_example": per_example}, f, ensure_ascii=False, indent=2)
    print(f"\nSaved detailed results to {args.output}")


if __name__ == "__main__":
    main()
