#!/usr/bin/env python
"""
Huggingface transformers model downloader.
"""
import logging
import os

import requests
from tqdm import tqdm

from transformers import AutoConfig
from transformers.configuration_utils import CONFIG_NAME
from transformers.file_utils import hf_bucket_url, WEIGHTS_NAME, TF2_WEIGHTS_NAME, http_get
from transformers.tokenization_auto import TOKENIZER_MAPPING
from transformers.tokenization_utils_base import SPECIAL_TOKENS_MAP_FILE, ADDED_TOKENS_FILE, \
    TOKENIZER_CONFIG_FILE, FULL_TOKENIZER_FILE

logger = logging.getLogger(__name__)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Huggingface model downloader")
    parser.add_argument('-s', '--short_name', required=True,
                        help='model short name, like bert-base-chinese')
    parser.add_argument('-o', '--output_dir', required=False,
                        help='optional, output directory to save model files, default to the model short name')
    parser.add_argument('-t', '--type', required=False, default='pt',
                        help='optional, type of model, pt for torch and tf for tensorflow, default to pt')
    parser.add_argument('-l', '--list', required=False, action='store_true',
                        help='optional, list urls only, default to False')
    args = parser.parse_args()
    if not args.output_dir:
        args.output_dir = f'{args.short_name}-cached'
    if not (args.type == 'pt' or args.type == 'tf'):
        raise ValueError(f'{args.type} is not a valid type name, choose one from [tf, pt]')
    return args


def url_is_alive(url):
    request = requests.head(url)
    return True if request.status_code == 200 else False


def config_file_from_short_name(short_name):
    return hf_bucket_url(short_name, filename=CONFIG_NAME, use_cdn=False)


def tokenizer_files_from_short_name(short_name):
    """Get all possible files for a tokenizer model by short name"""
    use_fast = False
    config = AutoConfig.from_pretrained(short_name)
    vocab_files = []
    for config_class, (tokenizer_class_py, tokenizer_class_fast) in TOKENIZER_MAPPING.items():
        if isinstance(config, config_class):
            tokenizer_class = tokenizer_class_fast if (use_fast and tokenizer_class_fast) else tokenizer_class_py
            vocab_files = list(tokenizer_class.vocab_files_names.values())
    additional_files = [ADDED_TOKENS_FILE, SPECIAL_TOKENS_MAP_FILE, TOKENIZER_CONFIG_FILE, FULL_TOKENIZER_FILE]
    tokenizer_files = []
    for filename in vocab_files + additional_files:
        tokenizer_files.append(hf_bucket_url(short_name, filename=filename, use_cdn=False))
    return tokenizer_files


def model_file_from_short_name(short_name, model_type):
    """Get model weights file by short name"""
    model_file = hf_bucket_url(
        short_name,
        filename=(TF2_WEIGHTS_NAME if model_type == 'tf' else WEIGHTS_NAME),
        use_cdn=True
    )
    return model_file


def download_url(url, local_file):
    """Download a url to a local file with a progress bar"""
    response = requests.get(url, stream=True)
    content_length = response.headers.get("Content-Length")
    total = 0 + int(content_length) if content_length is not None else None
    progress = tqdm(unit="B", unit_scale=True, total=total, initial=0, desc="Downloading",
                    disable=bool(logger.getEffectiveLevel() == logging.NOTSET))
    with open(local_file, 'wb') as fout:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                progress.update(len(chunk))
                fout.write(chunk)
    progress.close()


def download_all(files, list_only, output_dir):
    if list_only:
        for filename in files:
            print(filename)
    else:
        os.makedirs(output_dir, exist_ok=True)
        for url in files:
            print(url)
            if url_is_alive(url):
                local_file = os.path.join(output_dir, os.path.basename(url))
                download_url(url, local_file)
                print('[OK]')
            else:
                print('[NOT FOUND]')


def main():
    args = parse_args()
    config_file = config_file_from_short_name(args.short_name)
    tokenizer_files = tokenizer_files_from_short_name(args.short_name)
    model_file = model_file_from_short_name(args.short_name, args.type)
    download_all([config_file] + tokenizer_files + [model_file], args.list, args.output_dir)


if __name__ == '__main__':
    main()
