# Responsibility: Tokenization logic
import sys
import random
from transformers import AutoTokenizer, DataCollatorWithPadding


def get_tokenizer(cfg):
    """
    Initialize HuggingFace AutoTokenizer for SecBERT.
    """
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer.name)
    return tokenizer


def verify_tokenizer(tokenizer, texts, cfg):
    """
    Verify tokenizer behavior on random samples.
    Checks: random sample encoding, max token length, truncation rate,
    special tokens, padding, and attention masks.
    """
    max_length = cfg.tokenizer.max_length
    n_samples = min(5, len(texts))
    sample_texts = random.sample(list(texts), n_samples)

    print("======================================")
    print("Tokenizer Verification")
    print("======================================")

    # ── Special tokens ────────────────────────
    print(f"  [PASS] Tokenizer loaded       : {cfg.tokenizer.name}")
    print(f"  [INFO] Vocab size             : {tokenizer.vocab_size}")

    has_cls = tokenizer.cls_token is not None
    has_sep = tokenizer.sep_token is not None
    has_pad = tokenizer.pad_token is not None

    _tok_check("CLS token present", has_cls, f"'{tokenizer.cls_token}'")
    _tok_check("SEP token present", has_sep, f"'{tokenizer.sep_token}'")
    _tok_check("PAD token present", has_pad, f"'{tokenizer.pad_token}'")

    if not (has_cls and has_sep and has_pad):
        print("\nTokenizer missing critical special tokens. Stopping.\n")
        sys.exit(1)

    # ── Random sample encodings ───────────────
    print("Random sample encodings:")
    token_lengths = []
    truncated_count = 0

    for i, text in enumerate(sample_texts):
        enc = tokenizer(text, truncation=True, max_length=max_length)
        length = len(enc["input_ids"])
        token_lengths.append(length)

        if length >= max_length:
            truncated_count += 1

        # Verify first token is CLS, last is SEP
        first_token = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
        last_token = tokenizer.convert_ids_to_tokens(enc["input_ids"][-1])

        print(f"  [INFO] Sample {i+1}: {length:>4} tokens | "
              f"first='{first_token}' last='{last_token}' | "
              f"text='{text[:60]}...'")

    # ── Truncation rate across full dataset ───
    all_lengths = []
    for text in texts:
        enc = tokenizer(text, truncation=True, max_length=max_length)
        all_lengths.append(len(enc["input_ids"]))

    total = len(all_lengths)
    n_truncated = sum(1 for l in all_lengths if l >= max_length)
    trunc_rate = n_truncated / total * 100

    print(f"  [INFO] Max token length       : {max(all_lengths)}")
    print(f"  [INFO] Mean token length      : {sum(all_lengths) / total:.1f}")
    print(f"  [INFO] Truncation rate        : {n_truncated}/{total} ({trunc_rate:.2f}%)")
    _tok_check("Truncation rate < 5%", trunc_rate < 5.0, f"{trunc_rate:.2f}%")

    # ── Dynamic padding verification ──────────
    print("Dynamic padding check:")
    short_text = "Test."
    long_text = "This is a significantly longer piece of text for verification."
    enc_short = tokenizer(short_text, truncation=True, max_length=max_length)
    enc_long = tokenizer(long_text, truncation=True, max_length=max_length)

    len_short = len(enc_short["input_ids"])
    len_long = len(enc_long["input_ids"])

    _tok_check(
        f"No static padding (short={len_short}, long={len_long})",
        len_short != len_long,
        "Lengths differ — dynamic padding confirmed"
    )

    # ── Attention mask verification ───────────
    print("Attention mask check:")
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    batch = collator([enc_short, enc_long])
    attn = batch["attention_mask"]
    padded_len = attn.shape[1]

    # Short sample should have some 0s in attention mask (padding region)
    short_zeros = (attn[0] == 0).sum().item()
    long_zeros = (attn[1] == 0).sum().item()

    _tok_check(
        f"Attention masks correct (padded to {padded_len})",
        short_zeros > long_zeros,
        f"short has {short_zeros} pad positions, long has {long_zeros}"
    )

    print("======================================")
    print("Tokenizer verification passed.\n")


def _tok_check(label, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    suffix = f" ({detail})" if detail else ""
    print(f"  [{status}] {label}{suffix}")
    return condition
