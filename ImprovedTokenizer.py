import numpy as np
from collections import defaultdict
import re
import logging

logger = logging.getLogger("AdvancedNewsFactor.ImprovedTokenizer")

class ImprovedTokenizer:
    """Advanced tokenization system with BERT-like tokenization"""
    
    def __init__(self, vocab_size=50000):
        self.vocab_size = vocab_size
        self.word_to_idx = {'[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3}
        self.idx_to_word = {0: '[PAD]', 1: '[UNK]', 2: '[CLS]', 3: '[SEP]'}
        self.vocab_count = 4
        self.word_freq = defaultdict(int)
        
    def build_vocab(self, texts):
        for text in texts:
            words = self.tokenize_text(text)
            for word in words:
                self.word_freq[word] += 1
        
        sorted_words = sorted(self.word_freq.items(), key=lambda x: x[1], reverse=True)
        
        for word, freq in sorted_words[:self.vocab_size - 4]:
            if word not in self.word_to_idx:
                self.word_to_idx[word] = self.vocab_count
                self.idx_to_word[self.vocab_count] = word
                self.vocab_count += 1
    
    def tokenize_text(self, text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        words = text.strip().split()
        processed_words = []
        for word in words:
            if len(word) > 6:
                processed_words.extend(self.split_word(word))
            else:
                processed_words.append(word)
        
        return processed_words
    
    def split_word(self, word):
        common_prefixes = ['un', 'pre', 'dis', 'mis', 're', 'over', 'under', 'out']
        common_suffixes = ['ing', 'ed', 'er', 'est', 'ly', 'tion', 'sion', 'ness', 'ment']
        
        parts = [word]
        
        for prefix in common_prefixes:
            if word.startswith(prefix) and len(word) > len(prefix) + 2:
                rest = word[len(prefix):]
                parts = [prefix, rest]
                word = rest
                break
        
        if len(parts) > 1:
            word = parts[-1]
        
        for suffix in common_suffixes:
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                base = word[:-len(suffix)]
                if len(parts) > 1:
                    parts[-1] = base
                    parts.append(suffix)
                else:
                    parts = [base, suffix]
                break
        
        return parts
    
    def encode(self, text, max_length=100, add_special_tokens=True):
        words = self.tokenize_text(text)
        
        tokens = []
        if add_special_tokens:
            tokens.append(self.word_to_idx['[CLS]'])
        
        for word in words:
            if word in self.word_to_idx:
                tokens.append(self.word_to_idx[word])
            else:
                tokens.append(self.word_to_idx['[UNK]'])
        
        if add_special_tokens:
            tokens.append(self.word_to_idx['[SEP]'])
        
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
        else:
            tokens.extend([self.word_to_idx['[PAD]']] * (max_length - len(tokens)))
        
        tokens = np.array(tokens)
        if tokens.ndim == 1:
            tokens = tokens.reshape(1, -1)

        return tokens