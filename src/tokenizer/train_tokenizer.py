import argparse
import glob
import os

from tokenizers import ByteLevelBPETokenizer

from src.utils.config import load_config
from src.utils.logging_utils import get_logger

log = get_logger(__name__)


def main(cfg_path: str):
    cfg = load_config(cfg_path)
    corpus_dir = cfg['corpus_dir']
    files = sorted(glob.glob(os.path.join(corpus_dir, '**', '*.txt'), recursive=True))
    if not files:
        raise FileNotFoundError(f'no .txt files found under {corpus_dir} -- populate the corpus before training a tokenizer (see data/README.md)')
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(files=files, vocab_size=cfg['vocab_size'], special_tokens=cfg.get('special_tokens', ['<pad>', '<bos>', '<eos>', '<unk>']))
    out_path = cfg['out_path']
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tokenizer.save(out_path)
    log.info('trained on %d files, vocab_size=%d -> %s', len(files), cfg['vocab_size'], out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/tokenizer.yaml')
    args = parser.parse_args()
    main(args.config)
