import os
import time
import asyncio
import pandas as pd
from datasets import load_dataset
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable
import random

# ==========================================
# 1. Configuration
# ==========================================
GEMINI_API_KEY = "GEMINI_API_KEY_HERE"
GEMINI_MODEL = "gemini-3-flash-preview"
BATCH_SIZE = 50  # Reduced to avoid rate limits
PARALLEL_REQUESTS = 20  # Reduced to be safer
MAX_RETRIES = 5
INITIAL_BACKOFF = 2  # seconds

CHECKPOINT_FILE = "progress_checkpoint.csv"
OUTPUT_FILE = "ghana_entities_labeled.csv"

genai.configure(api_key=GEMINI_API_KEY)

# ==========================================
# 2. Load the Dataset
# ==========================================
print("Loading dataset from Hugging Face...")
ds = load_dataset("ghananlpcommunity/ghana-named-entities", split="train")

df = ds.to_pandas().reset_index(drop=True)
print(f"Dataset loaded: {len(df)} rows, columns: {list(df.columns)}")

# ==========================================
# 3. Processing Functions
# ==========================================
def build_prompt(batch_df):
    lines = [str(row['phrase']).strip() for _, row in batch_df.iterrows()]
    prompt_text = "\n".join(lines)
    return f"""You are an expert in Ghanaian culture and context. I am providing a list of {len(batch_df)} entities.
Determine if each entity is specific to the Ghanaian local context (e.g., 'Accra', 'NDC', 'Waakye', 'Kofi').
Mark as False if the term is universal/generic (e.g., 'Monday', 'Parliament', 'Ministry') or clearly non-Ghanaian.

Output EXACTLY {len(batch_df)} lines.
Each line must contain ONLY the word 'True' or 'False' in the exact same order as the input.
Do not output numbers, explanations, blank lines, or any other text.

--- Example ---
Input:
Ghana
Parliament
Accra
Monday

Output:
True
False
True
False
---------------

Input Data:
{prompt_text}"""


def parse_response(response_text, expected_count):
    raw_responses = [line.strip().lower() for line in response_text.split('\n') if line.strip()]
    flags_raw = [val for val in raw_responses if val in ('true', 'false')]
    flags = [val == 'true' for val in flags_raw]
    if len(flags) != expected_count:
        raise ValueError(f"Alignment mismatch: expected {expected_count}, got {len(flags)}. Snippet: {response_text[:300]}")
    return flags


async def get_model_predictions_with_backoff(model, batch_df, semaphore):
    """Make API call with exponential backoff for rate limiting."""
    async with semaphore:
        prompt = build_prompt(batch_df)
        expected_count = len(batch_df)
        
        for attempt in range(MAX_RETRIES):
            try:
                # Run the synchronous Gemini call in a thread pool
                loop = asyncio.get_running_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: model.generate_content(
                        prompt,
                        generation_config=genai.GenerationConfig(
                            temperature=0.1,
                            max_output_tokens=4096,
                        )
                    )
                )
                
                # Check if response was blocked
                if not response.text:
                    raise ValueError("Empty response - possibly blocked")
                
                return parse_response(response.text.strip(), expected_count)
                
            except ResourceExhausted as e:
                # Rate limit hit - use exponential backoff with jitter
                wait_time = min(INITIAL_BACKOFF * (2 ** attempt) + random.uniform(0, 1), 60)
                print(f"\n[Rate Limit] Attempt {attempt + 1}/{MAX_RETRIES}. Waiting {wait_time:.1f}s...")
                await asyncio.sleep(wait_time)
                
            except ServiceUnavailable as e:
                # Service temporarily unavailable
                wait_time = min(INITIAL_BACKOFF * (2 ** attempt), 30)
                print(f"\n[Service Unavailable] Attempt {attempt + 1}/{MAX_RETRIES}. Waiting {wait_time:.1f}s...")
                await asyncio.sleep(wait_time)
                
            except Exception as e:
                print(f"\n[Error] Attempt {attempt + 1}/{MAX_RETRIES}: {e}")
                if attempt == MAX_RETRIES - 1:
                    raise  # Re-raise on final attempt
                await asyncio.sleep(2)
        
        raise RuntimeError("Max retries exceeded")


async def process_single_batch(model, batch_df, semaphore):
    """Process a single batch with retry logic that splits on failure."""
    try:
        return await get_model_predictions_with_backoff(model, batch_df, semaphore)
    except Exception as e:
        print(f"\n[Warning] Batch failed after retries: {e}. Splitting into smaller chunks...")
        
        # Split into 3 sub-batches and process sequentially
        n = len(batch_df)
        if n <= 5:
            # Too small to split, mark as failed
            print(f"  -> Batch too small ({n} items), cannot split further.")
            return None
            
        split_size = (n + 2) // 3
        sub_batches = [batch_df.iloc[i: i + split_size] for i in range(0, n, split_size)]
        
        results = []
        for i, sub_batch in enumerate(sub_batches):
            await asyncio.sleep(3)  # Delay between sub-batches
            try:
                sub_flags = await get_model_predictions_with_backoff(model, sub_batch, semaphore)
                results.extend(sub_flags)
                print(f"  -> Sub-batch {i+1}/3 succeeded ({len(sub_batch)} items)")
            except Exception as sub_e:
                print(f"  -> Sub-batch {i+1}/3 failed: {sub_e}.")
                return None  # Signal complete failure
        
        return results


async def process_all_batches(model, batches_to_process):
    """
    Process all batches with proper concurrency control.
    Uses a queue-based approach with limited workers.
    """
    semaphore = asyncio.Semaphore(PARALLEL_REQUESTS)
    results = [None] * len(batches_to_process)
    failed_batches = []  # Track batches that need retry
    
    async def worker(idx, batch_df):
        flags = await process_single_batch(model, batch_df, semaphore)
        if flags is None:
            failed_batches.append((idx, batch_df))
        else:
            results[idx] = (batch_df, flags)
    
    # Create tasks for all batches
    tasks = [worker(i, batch) for i, batch in enumerate(batches_to_process)]
    await asyncio.gather(*tasks)
    
    # Retry failed batches one at a time with longer delays
    if failed_batches:
        print(f"\n[Retry Phase] Processing {len(failed_batches)} failed batches sequentially...")
        for idx, batch_df in failed_batches:
            await asyncio.sleep(5)  # Longer delay between retries
            try:
                flags = await get_model_predictions_with_backoff(model, batch_df, asyncio.Semaphore(1))
                results[idx] = (batch_df, flags)
                print(f"  -> Retry succeeded for batch at index {idx}")
            except Exception as e:
                print(f"  -> Retry failed for batch at index {idx}: {e}. Will need manual review.")
                # Don't write these to checkpoint - they'll be reprocessed on next run
    
    # Filter out None results (failed batches that weren't recovered)
    valid_results = [r for r in results if r is not None]
    return valid_results


# ==========================================
# 4. Resume State & Initialization
# ==========================================
if os.path.exists(CHECKPOINT_FILE):
    progress_df = pd.read_csv(CHECKPOINT_FILE)
    # Ensure we only count valid entries (not None/NaN)
    processed_indices = set(progress_df['original_index'].dropna().astype(int).tolist())
    print(f"Resuming from checkpoint: {len(processed_indices)} rows already processed.")
else:
    processed_indices = set()
    # Initialize empty files with headers
    pd.DataFrame(columns=['original_index', 'keep']).to_csv(CHECKPOINT_FILE, index=False)
    pd.DataFrame(columns=list(df.columns) + ['label']).to_csv(OUTPUT_FILE, index=False)
    print("No checkpoint found. Starting fresh.")

# ==========================================
# 5. Main Processing Loop - Single Event Loop
# ==========================================
print(f"\nProcessing {len(df)} rows | batch size: {BATCH_SIZE} | parallel requests: {PARALLEL_REQUESTS}")

# Build the list of all unprocessed batches - FIXED to handle partial batches correctly
all_batches = []
current_batch = []
batch_indices = []  # Track indices for the current batch

for idx in range(len(df)):
    if idx in processed_indices:
        continue  # Skip already processed rows
    
    current_batch.append(idx)
    
    # When batch is full or at end, add it
    if len(current_batch) >= BATCH_SIZE or idx == len(df) - 1:
        if current_batch:  # Only add if not empty
            batch_df = df.iloc[current_batch]
            all_batches.append(batch_df)
            current_batch = []

print(f"Batches remaining: {len(all_batches)} (covering {sum(len(b) for b in all_batches)} rows)")

if len(all_batches) == 0:
    print("All batches already processed!")
else:
    # Create model instance here to reuse across calls
    model = genai.GenerativeModel(GEMINI_MODEL)
    
    # Process all batches in a single event loop
    async def main():
        # Process in chunks to allow for periodic checkpointing
        chunk_size = PARALLEL_REQUESTS * 2  # Process 2 waves at a time
        
        for chunk_start in range(0, len(all_batches), chunk_size):
            chunk = all_batches[chunk_start: chunk_start + chunk_size]
            
            print(f"\n{'='*60}")
            print(f"Processing chunk {chunk_start // chunk_size + 1}/{(len(all_batches) + chunk_size - 1) // chunk_size}")
            print(f"({len(chunk)} batches covering ~{sum(len(b) for b in chunk)} rows)")
            print(f"{'='*60}")
            
            # Process this chunk
            results = await process_all_batches(model, chunk)
            
            # Write results
            for batch_df, flags in results:
                # Checkpoint
                checkpoint_rows = pd.DataFrame({
                    'original_index': batch_df.index.tolist(),
                    'keep': flags
                })
                checkpoint_rows.to_csv(CHECKPOINT_FILE, mode='a', header=False, index=False)
                
                # Output
                labeled_batch = batch_df.copy()
                labeled_batch['label'] = ['Ghanaian' if f else 'Non-Ghanaian' for f in flags]
                labeled_batch.to_csv(OUTPUT_FILE, mode='a', header=False, index=False)
                
                print(f"  -> Rows {batch_df.index[0]}–{batch_df.index[-1]} | "
                      f"Ghanaian: {sum(flags)}, Non-Ghanaian: {len(flags) - sum(flags)}")
            
            # Brief pause between chunks to be nice to the API
            if chunk_start + chunk_size < len(all_batches):
                print(f"\n[Pause] 5 seconds before next chunk...")
                await asyncio.sleep(5)
    
    # Run the entire async program with a single event loop
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n[Interrupted] Progress saved to checkpoint. Resume anytime by running again.")
    except Exception as e:
        print(f"\n\n[Fatal Error] {e}")
        raise

print(f"\n{'='*60}")
print(f"Done! Output saved to: {OUTPUT_FILE}")
print(f"Checkpoint saved to: {CHECKPOINT_FILE}")
print(f"{'='*60}")
