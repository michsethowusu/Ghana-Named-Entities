#!/usr/bin/env python3
"""
Generate descriptions for noun phrases using Mistral API.
Asks model to explain what/who each entity is (from Ghanaian news context).

Improvements over v1:
  - Uses ENTITY | DESC format so matching is name-based (not position-based)
  - Validates that description seems to be about the right entity
  - Detects mismatches and retries them in smaller batches
  - Full retry loop with exponential backoff on API errors
"""

import pandas as pd
import time
import re
import os
from pathlib import Path
from difflib import SequenceMatcher
from openai import OpenAI

# ─── CONFIG ──────────────────────────────────────────────────────────────────

MISTRAL_API_KEY = "MISTRAL_API_KEY_HERE"  # <-- Replace with your actual Mistral API key
MODEL          = "mistralai/mistral-small-3.1-24b-instruct-2503"  # Options: mistral-large-latest, mistral-medium, mistral-small, etc.

INPUT_FILE     = "/media/owusus/Godstestimo/NLP-Projects/GhanaNamedEntities/data/part6-mich_named_entities_news.csv"
OUTPUT_FILE    = "/media/owusus/Godstestimo/NLP-Projects/GhanaNamedEntities/output/part6-mich_named_entities_with_descriptions.csv"

BATCH_SIZE     = 100   # Normal batch size
RETRY_BATCH    = 10    # Batch size for first retry
SOLO_BATCH     = 1     # Batch size for final retry (one at a time)
API_DELAY      = 2     # Seconds between batches
MAX_API_TRIES  = 3     # Max attempts per API call before giving up

# A description is considered valid if at least one word from the phrase
# (longer than 3 chars) appears in the description text.
VALIDATION_MIN_WORD_LEN = 3

# ─── CLIENT ──────────────────────────────────────────────────────────────────

client = OpenAI(
    base_url="https://api.mistral.ai/v1",
    api_key=MISTRAL_API_KEY,
)

# ─── HELPERS ─────────────────────────────────────────────────────────────────

def format_duration(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def fuzzy_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def is_likely_correct(phrase: str, description: str) -> bool:
    """
    Returns True if the description appears to be about the right entity.
    Checks that at least one significant word from the phrase appears in
    the description, OR that the phrase name appears (fuzzy) in the description.
    """
    if not description or description == "[ERROR]":
        return False

    desc_lower = description.lower()
    phrase_lower = phrase.lower()

    # Direct substring match (most reliable)
    if phrase_lower in desc_lower:
        return True

    # Check individual significant words
    words = [w for w in re.split(r'\W+', phrase_lower) if len(w) > VALIDATION_MIN_WORD_LEN]
    if not words:
        return True  # Too short to validate meaningfully

    return any(w in desc_lower for w in words)


# ─── API CALL ────────────────────────────────────────────────────────────────

def call_api(prompt: str) -> str | None:
    """Call the Mistral API with retry/backoff. Returns raw text or None."""
    for attempt in range(1, MAX_API_TRIES + 1):
        try:
            completion = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                top_p=0.9,
                max_tokens=4096,
                stream=False,
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            wait = 2 ** attempt
            print(f"      API error (attempt {attempt}/{MAX_API_TRIES}): {e}. Retrying in {wait}s...")
            time.sleep(wait)
    return None


# ─── BATCH DESCRIPTION GENERATION ────────────────────────────────────────────

def generate_descriptions_batch(phrases: list[str]) -> dict[str, str]:
    """
    Send a batch of phrase strings to the model.
    Returns a dict of {phrase: description} for successfully parsed entries.
    Uses ENTITY | DESC format so matching is name-based, not position-based.
    """
    if not phrases:
        return {}

    phrases_block = "\n".join(phrases)

    prompt = f"""These named entities were extracted from Ghanaian news articles.
For each entity below, write a brief description explaining what it is or who they are.

Use EXACTLY this format — one entry per line, no extra lines:
ENTITY: <exact entity name> | DESC: <description that mentions the entity name>

EXAMPLE OUTPUT:
ENTITY: Parliament | DESC: Parliament is the national legislative body of Ghana where laws are made by elected representatives.
ENTITY: Nana Akufo-Addo | DESC: Nana Akufo-Addo is the former President of Ghana who served two terms leading the country's executive branch.
ENTITY: Bank of Ghana | DESC: Bank of Ghana is the central bank responsible for managing the country's currency and monetary policy.

ENTITIES TO DESCRIBE:
{phrases_block}

OUTPUT:"""

    raw = call_api(prompt)
    if not raw:
        return {}

    result: dict[str, str] = {}

    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        # Match: ENTITY: <name> | DESC: <description>
        m = re.match(r'ENTITY\s*:\s*(.+?)\s*\|\s*DESC\s*:\s*(.+)', line, re.IGNORECASE)
        if m:
            entity = m.group(1).strip().strip('"\'')
            desc   = m.group(2).strip().strip('"\'')
            result[entity] = desc

    return result


def match_results_to_phrases(phrases: list[str], raw_map: dict[str, str]) -> dict[str, str]:
    """
    Match the model's returned entity keys back to the original phrases.
    Tries exact match first, then fuzzy match (ratio > 0.8).
    Returns {phrase: description} for all that were matched.
    """
    matched: dict[str, str] = {}
    used_keys: set[str] = set()

    for phrase in phrases:
        # 1. Exact match
        if phrase in raw_map:
            matched[phrase] = raw_map[phrase]
            used_keys.add(phrase)
            continue

        # 2. Case-insensitive exact
        for key in raw_map:
            if key.lower() == phrase.lower() and key not in used_keys:
                matched[phrase] = raw_map[key]
                used_keys.add(key)
                break
        else:
            # 3. Fuzzy match (best ratio above threshold)
            best_key, best_score = None, 0.0
            for key in raw_map:
                if key in used_keys:
                    continue
                score = fuzzy_ratio(phrase, key)
                if score > best_score:
                    best_key, best_score = key, score

            if best_key and best_score >= 0.8:
                matched[phrase] = raw_map[best_key]
                used_keys.add(best_key)

    return matched


# ─── PROCESS WITH RETRY ───────────────────────────────────────────────────────

def process_batch_with_retry(phrases: list[str]) -> dict[str, str]:
    """
    Process a list of phrases with up to 3 rounds of retries for failures:
      Round 1: Full batch (up to BATCH_SIZE)
      Round 2: Failed phrases in groups of RETRY_BATCH
      Round 3: Still-failed phrases one at a time
    Returns {phrase: description} for all successfully described phrases.
    """
    final: dict[str, str] = {}

    def run_round(batch: list[str], label: str) -> list[str]:
        """Run one round; return list of phrases that still need processing."""
        still_failed: list[str] = []

        # Split into sub-batches if needed
        size = len(batch)
        for i in range(0, size, size):  # single chunk for this round
            sub = batch[i : i + size]
            raw_map  = generate_descriptions_batch(sub)
            matched  = match_results_to_phrases(sub, raw_map)

            for phrase in sub:
                desc = matched.get(phrase)
                if desc and is_likely_correct(phrase, desc):
                    final[phrase] = desc
                else:
                    still_failed.append(phrase)
                    if desc:
                        print(f"      ⚠  Mismatch detected [{label}]: '{phrase}' → '{desc[:60]}...'")
                    else:
                        print(f"      ✗  Not matched [{label}]: '{phrase}'")

        return still_failed

    # ── Round 1: full batch ──────────────────────────────────────────────────
    failed = []
    raw_map = generate_descriptions_batch(phrases)
    matched = match_results_to_phrases(phrases, raw_map)

    for phrase in phrases:
        desc = matched.get(phrase)
        if desc and is_likely_correct(phrase, desc):
            final[phrase] = desc
        else:
            failed.append(phrase)
            if desc:
                print(f"      ⚠  Mismatch [round 1]: '{phrase}' → '{desc[:60]}'")
            else:
                print(f"      ✗  Not matched [round 1]: '{phrase}'")

    if not failed:
        return final

    print(f"      → {len(failed)} phrases need retry (batch of {RETRY_BATCH})...")
    time.sleep(API_DELAY)

    # ── Round 2: smaller batches ─────────────────────────────────────────────
    still_failed: list[str] = []
    for i in range(0, len(failed), RETRY_BATCH):
        sub = failed[i : i + RETRY_BATCH]
        raw_map2 = generate_descriptions_batch(sub)
        matched2 = match_results_to_phrases(sub, raw_map2)

        for phrase in sub:
            desc = matched2.get(phrase)
            if desc and is_likely_correct(phrase, desc):
                final[phrase] = desc
            else:
                still_failed.append(phrase)
                if desc:
                    print(f"      ⚠  Mismatch [round 2]: '{phrase}' → '{desc[:60]}'")

        if i + RETRY_BATCH < len(failed):
            time.sleep(API_DELAY)

    if not still_failed:
        return final

    print(f"      → {len(still_failed)} phrases need solo retry...")
    time.sleep(API_DELAY)

    # ── Round 3: one at a time ───────────────────────────────────────────────
    for phrase in still_failed:
        raw_map3 = generate_descriptions_batch([phrase])
        matched3 = match_results_to_phrases([phrase], raw_map3)
        desc = matched3.get(phrase)

        if desc and is_likely_correct(phrase, desc):
            final[phrase] = desc
            print(f"      ✓  Recovered [solo]: '{phrase}'")
        else:
            # Last resort: store whatever we got, or mark error
            final[phrase] = desc if desc else "[ERROR]"
            print(f"      ✗  Gave up [solo]: '{phrase}'")

        time.sleep(1)  # shorter delay for solo calls

    return final


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    # Load input
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"Error: Input file '{INPUT_FILE}' not found!")
        return

    print(f"Loaded {len(df)} noun phrases from {INPUT_FILE}")

    if 'phrase' not in df.columns:
        print("Error: 'phrase' column not found in CSV.")
        return

    # ── Resume logic ─────────────────────────────────────────────────────────
    start_idx = 0
    if Path(OUTPUT_FILE).exists():
        print(f"Found existing output file: {OUTPUT_FILE}")
        existing_df = pd.read_csv(OUTPUT_FILE)

        if 'description' in existing_df.columns:
            non_null = existing_df['description'].notna() & (existing_df['description'] != '')
            if non_null.any():
                start_idx = int(non_null.sum())
                print(f"Resuming from index {start_idx} ({start_idx} already done)")
                df = existing_df
            else:
                df['description'] = ''
        else:
            df['description'] = ''
    else:
        df['description'] = ''

    total_remaining = len(df) - start_idx
    total_batches   = (total_remaining + BATCH_SIZE - 1) // BATCH_SIZE
    process_start   = time.time()

    print(f"\nProcessing {total_remaining} phrases | batch={BATCH_SIZE} | retry={RETRY_BATCH} | solo=1")
    print("=" * 75)

    batch_num = 0
    idx = start_idx

    while idx < len(df):
        batch_end = min(idx + BATCH_SIZE, len(df))
        phrases   = df.iloc[idx:batch_end]['phrase'].tolist()

        batch_num += 1
        pct = (batch_num / total_batches) * 100
        preview = ", ".join(phrases[:3]) + ("..." if len(phrases) > 3 else "")
        print(f"\n[Batch {batch_num}/{total_batches}] [{pct:.1f}%] {preview}")

        # Process with retry
        results = process_batch_with_retry(phrases)

        # Write results back to df
        for i, phrase in enumerate(phrases):
            actual_idx = idx + i
            desc = results.get(phrase, "[ERROR]")
            df.at[actual_idx, 'description'] = desc

            status = "✓" if desc and desc != "[ERROR]" else "✗"
            print(f"    {status} {phrase[:50]}: {str(desc)[:70]}{'...' if desc and len(desc) > 70 else ''}")

        # Save after every batch
        df.to_csv(OUTPUT_FILE, index=False)

        elapsed = time.time() - process_start
        avg     = elapsed / batch_num
        eta     = avg * (total_batches - batch_num)
        print(f"    Progress: {batch_end - start_idx}/{total_remaining} | ETA: {format_duration(eta)}")

        idx = batch_end
        if idx < len(df):
            time.sleep(API_DELAY)

    # ── Summary ───────────────────────────────────────────────────────────────
    total_elapsed = time.time() - process_start
    successful    = df['description'].notna() & ~df['description'].isin(['', '[ERROR]'])
    errors        = df['description'] == '[ERROR]'

    print(f"\n{'=' * 75}")
    print(f"✓ Complete! Total time : {format_duration(total_elapsed)}")
    print(f"✓ Results saved to     : {OUTPUT_FILE}")
    print(f"✓ Successful           : {successful.sum()}/{len(df)} ({successful.sum()/len(df)*100:.1f}%)")
    print(f"✗ Errors/unmatched     : {errors.sum()}")


if __name__ == "__main__":
    main()
