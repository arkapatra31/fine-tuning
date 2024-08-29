import os
import pandas as pd
import pickle
from typing import List, Any
from importlib._bootstrap_external import path_separators
from langchain_text_splitters import TokenTextSplitter
from dotenv import load_dotenv
from datasets import Dataset
from transformers import DistilBertTokenizerFast

load_dotenv()

extracted_text_file_path = os.getenv("PDF_TEXT_EXTRACT")
base_slm: str = os.getenv("BASE_SLM")
dataset_save_path = os.getenv("SAVE_DATASET_LOCAL_PATH")

# Load the tokenizer from the pretrained base model
tokenizer = DistilBertTokenizerFast.from_pretrained(base_slm)


def chunkify_data(extract_file_path: str) -> List[str]:
    # Initialise the Token Splitter
    token_splitter = TokenTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len
    )

    chunks: List[str]
    with open(file=extract_file_path, mode="r", encoding="utf-8") as txt_file:
        contents = txt_file.read()
        chunks = token_splitter.split_text(contents)
    txt_file.close()
    return chunks


def create_dataset():
    chunks = chunkify_data(extracted_text_file_path)
    labels = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]

    # Tokenize the text chunks
    enocded_inputs = tokenizer(chunks, truncation=True, padding=True)
    dataset = Dataset.from_dict(enocded_inputs)
    dataset.add_column("labels", labels)
    dataset.save_to_disk(dataset_save_path)
    df = dataset.to_pandas()
    df.to_csv(dataset_save_path+"dataset.csv")
    return dataset


if __name__ == "__main__":
    create_dataset()
    dset = Dataset.load_from_disk(dataset_save_path)
    print(dset.data)
