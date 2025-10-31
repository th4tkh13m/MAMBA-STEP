import argparse
import logging
import os
import torch
import pandas as pd

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset
from functools import partial

def mask_pattern(tensor, start_pattern, end_pattern, ignore_index):
    """
    Masks the parts of the tensor that are not within the start and end patterns.

    Parameters:
        tensor (torch.Tensor): The tokenized input tensor.
        start_pattern (torch.Tensor): Tensor containing the start pattern.
        end_pattern (torch.Tensor): Tensor containing the end pattern.
        ignore_index (int): Value to use for indices that should be ignored.

    Returns:
        torch.Tensor: A tensor with masked values.
    """
    mask = torch.full(tensor.shape, ignore_index)
    start_pattern_length = len(start_pattern)
    end_pattern_length = len(end_pattern)

    i = 0
    in_sequence = False
    start_index = 0

    while i < len(tensor):
        if not in_sequence and i <= len(tensor) - start_pattern_length and torch.equal(tensor[i:i+start_pattern_length], start_pattern):
            in_sequence = True
            start_index = i + start_pattern_length
            i += start_pattern_length - 1
        elif in_sequence and i <= len(tensor) - end_pattern_length and torch.equal(tensor[i:i+end_pattern_length], end_pattern):
            mask[start_index: i+1] = tensor[start_index: i+1]
            in_sequence = False
            i += end_pattern_length - 1
        i += 1

    if in_sequence:
        mask[start_index:] = tensor[start_index:]
    return mask


def process_line(conversations, tokenizer, start_pattern, end_pattern, ignore_index):
    """
    Tokenizes a single conversation using the given tokenizer and applies masking.

    Parameters:
        conversations (list): List of conversation dictionaries.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer instance.
        start_pattern (torch.Tensor): The start pattern tensor.
        end_pattern (torch.Tensor): The end pattern tensor.
        ignore_index (int): Index value used for masking.

    Returns:
        dict: Dictionary containing tokenized input_ids, attention_mask, and labels.
    """
    templated_convs = tokenizer.apply_chat_template(conversations, tokenize=False)
    tokens = tokenizer(templated_convs, return_tensors="pt")
    input_ids = tokens['input_ids'].squeeze(dim=0)
    attention_mask = tokens['attention_mask'].squeeze(dim=0)
    labels = mask_pattern(input_ids, start_pattern, end_pattern, ignore_index)
    return {
        "input_ids": input_ids.tolist(),
        "attention_mask": attention_mask.tolist(),
        "labels": labels.tolist()
    }


def write_parquet_chunk(output_chunk, output_dir, part, total_parts):
    """
    Writes a chunk of tokenized data to a Parquet file.

    Parameters:
        output_chunk (list): Chunk of tokenized data.
        output_dir (str): Directory where the Parquet files will be saved.
        part (int): Index of the current chunk.
        total_parts (int): Total number of parts.
    """
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"train-{part}-of-{total_parts}.parquet")
    df = pd.DataFrame(output_chunk)
    df.to_parquet(file_path, index=False)
    logging.info(f"Saved {file_path}")


def split_and_write_parallel(output, output_dir, num_parts, max_workers):
    """
    Splits the tokenized output data into chunks and writes them to Parquet files in parallel.

    Parameters:
        output (list): List of tokenized output dictionaries.
        output_dir (str): Directory to save the Parquet files.
        num_parts (int): Number of parts to split the data into.
        max_workers (int): Number of threads to use for writing.
    """
    chunk_size = len(output) // num_parts
    chunks = [output[i * chunk_size:(i + 1) * chunk_size] for i in range(num_parts)]
    
    # Ensure any remaining elements are added to the last chunk.
    if len(output) % num_parts != 0:
        chunks[-1].extend(output[num_parts * chunk_size:])
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i, chunk in enumerate(chunks):
            part = i + 1
            futures.append(executor.submit(write_parquet_chunk, chunk, output_dir, part, num_parts))
        
        for future in tqdm(futures, desc="Writing Data in Parallel"):
            future.result()

# You need to change this accordingly for different datasets, since different datasets have different conversation field.
def build_conversations(dataset):
    """
    Constructs a conversation list from the dataset.

    Parameters:
        dataset (datasets.Dataset): Dataset loaded from Hugging Face.

    Returns:
        list: A list of conversations formatted as lists of message dictionaries.
    """
    conversations_output = []
    for dataitem in tqdm(dataset, desc="Building conversations"):
        conversations = dataitem["conversations"]
        conversation_items = []
        if len(conversations) != 2:
            logging.warning(f"Unexpected conversation length: {conversations}")
        for conversation in conversations:
            if conversation["from"] == "user":
                conversation_items.append({"role": "user", "content": conversation["value"]})
            elif conversation["from"] == "assistant":
                conversation_items.append({"role": "assistant", "content": conversation["value"]})
            else:
                # You need to change this accordingly for different datasets
                conversation_items.append(conversation)
        conversations_output.append(conversation_items)
    return conversations_output


def main(args):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    logging.info("Loading dataset...")
    dataset = load_dataset(args.dataset_name)[args.dataset_split]

    logging.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    
    start_pattern = torch.tensor(args.start_pattern, dtype=torch.long)
    end_pattern = torch.tensor([tokenizer.eos_token_id], dtype=torch.long)
    
    logging.info("Building conversation data...")
    conversations_output = build_conversations(dataset)
    total_conversations = len(conversations_output)
    logging.info(f"Total conversations: {total_conversations}")
    
    tokenized_output = []
    logging.info("Processing conversations in parallel...")

    # Process the conversations in parallel
    process_line_partial = partial(
        process_line,
        tokenizer=tokenizer,
        start_pattern=start_pattern,
        end_pattern=end_pattern,
        ignore_index=args.ignore_index
    )

    with ProcessPoolExecutor(max_workers=args.num_processes) as executor:
        futures = executor.map(process_line_partial, conversations_output, chunksize=args.chunksize)
        for result in tqdm(futures, total=total_conversations, desc="Tokenizing Data"):
            tokenized_output.append(result)

    logging.info("Writing tokenized data to Parquet files...")
    split_and_write_parallel(tokenized_output, args.output_dir, args.num_parts, args.num_threads)
    logging.info("Processing complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tokenize and process conversation data, then output tokenized results as Parquet files."
    )
    parser.add_argument("--dataset_name", type=str,
                        default="open-r1/OpenThoughts-114k-math",
                        help="Name of the dataset to load from Hugging Face.")
    parser.add_argument("--dataset_split", type=str,
                        default="train",
                        help="Dataset split to use (e.g., train, test).")
    parser.add_argument("--tokenizer_name", type=str,
                        default="meta-llama/Llama-3.2-3B-Instruct",
                        help="Name or path of the tokenizer to use.")
    parser.add_argument("--output_dir", type=str,
                        default="OpenThoughts2M_llama_math/tokenized_r1/",
                        help="Output directory for the Parquet files.")
    parser.add_argument("--num_processes", type=int,
                        default=64,
                        help="Number of processes for parallel tokenization.")
    parser.add_argument("--num_parts", type=int,
                        default=128,
                        help="Number of parts to split the tokenized data for output files.")
    parser.add_argument("--num_threads", type=int,
                        default=64,
                        help="Number of threads for parallel writing of Parquet files.")
    parser.add_argument("--chunksize", type=int,
                        default=100,
                        help="Chunksize for the process pool executor.")
    parser.add_argument("--start_pattern", type=int, nargs='+',
                        default=[78191, 128007, 271],
                        help="Token IDs to start mask. For llama, it corresponds to <Assistant>. For deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B, you need to set this to [151645]")
    parser.add_argument("--ignore_index", type=int,
                        default=-100,
                        help="Value to use for ignored indices during label masking.")

    args = parser.parse_args()
    main(args)
