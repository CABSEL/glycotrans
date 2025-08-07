import torch
import json
from transformers.models.bart.modeling_bart import shift_tokens_right
from torch.utils.data import Dataset

class GlycoBertTokenizer:
    def __init__(self, vocab_list, max_seq_length=512):
        # BERT's special tokens
        self.special_tokens = {
            'pad_token': '[PAD]',
            'cls_token': '[CLS]',
            'sep_token': '[SEP]',
            'unk_token': '[UNK]',
            'mask_token': '[MASK]'
        }
        
        # List of special token symbols
        special_token_symbols = list(self.special_tokens.values())

        # Filter out special tokens from vocab_list to prevent duplicates
        vocab_list = [word for word in vocab_list if word not in special_token_symbols]

        # Create a combined list of special tokens and vocab_list
        combined_list = special_token_symbols + vocab_list

        # Create vocab and reverse vocab dictionaries
        self.vocab = {word: idx for idx, word in enumerate(combined_list)}
        self.reverse_vocab = {idx: word for word, idx in self.vocab.items()}
        self.max_seq_length = max_seq_length

    def tokenize(self, text):
        return text.split()

    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]

        batch_token_ids = []
        batch_attention_masks = []
    
        for text in texts:
            tokens = self.tokenize(text)
            token_ids = [self.vocab.get(token, self.vocab[self.special_tokens['unk_token']]) for token in tokens]

            # Prepend [CLS] token and append [SEP] token
            token_ids = [self.vocab[self.special_tokens['cls_token']]] + token_ids + [self.vocab[self.special_tokens['sep_token']]]

            # Create attention mask
            attention_mask = [1] * len(token_ids)
            
            # Padding or truncating to the max_seq_length
            if len(token_ids) < self.max_seq_length:
                padding_length = self.max_seq_length - len(token_ids)
                token_ids += [self.vocab[self.special_tokens['pad_token']]] * padding_length
                attention_mask += [0] * padding_length
            else:
                token_ids = token_ids[:self.max_seq_length]
                attention_mask = attention_mask[:self.max_seq_length]

            batch_token_ids.append(torch.tensor(token_ids))
            batch_attention_masks.append(torch.tensor(attention_mask))

        return {
            "token_ids": torch.stack(batch_token_ids),
            "attention_mask": torch.stack(batch_attention_masks)
        }

    def decode(self, batch_token_ids, skip_special_tokens=False):
        if batch_token_ids.dim() == 1:
            batch_token_ids = batch_token_ids.unsqueeze(0)

        decoded_texts = []
        for token_ids in batch_token_ids:
            if skip_special_tokens:
                decoded_texts.append(' '.join([self.reverse_vocab[token_id.item()] for token_id in token_ids if token_id.item() not in [self.vocab[val] for val in self.special_tokens.values()]]))
            else:
                decoded_texts.append(' '.join([self.reverse_vocab[token_id.item()] for token_id in token_ids if token_id.item() != self.vocab[self.special_tokens['pad_token']]]))

        return decoded_texts if len(decoded_texts) > 1 else decoded_texts[0]

    
    def save_vocabulary(self, path="vocab.json"):
        with open(path, 'w') as file:
            json.dump(self.vocab, file)

    @property
    def vocab_size(self):
        """Returns the size of the vocabulary."""
        return len(self.vocab)

    @classmethod
    def load_vocabulary(cls, path="vocab.json", max_seq_length=512):
        with open(path, 'r') as file:
            loaded_vocab = json.load(file)
        return cls(list(loaded_vocab.keys()), max_seq_length) 
    

class GlycoBartTokenizer:
    def __init__(self, vocab_list, max_seq_length=512):
        # Special tokens
        self.special_tokens = {
            'pad_token': '<pad>',
            'bos_token': '<s>',
            'eos_token': '</s>',
            'sep_token': '<sep>',
            'cls_token': '<cls>',
            'unk_token': '<unk>',
            'mask_token': '<mask>'
        }
        
        # List of special token symbols
        special_token_symbols = list(self.special_tokens.values())

        # Filter out special tokens from vocab_list to prevent duplicates
        vocab_list = [word for word in vocab_list if word not in special_token_symbols]

        # Create a combined list of special tokens and vocab_list
        combined_list = special_token_symbols + vocab_list

        # Create vocab and reverse vocab dictionaries
        self.vocab = {word: idx for idx, word in enumerate(combined_list)}
        self.reverse_vocab = {idx: word for word, idx in self.vocab.items()}
        self.max_seq_length = max_seq_length

    def tokenize(self, text):
        return text.split()

    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]

        batch_token_ids = []
        batch_attention_masks = []
    
        for text in texts:
            tokens = self.tokenize(text)  # This will now always be a string
            token_ids = [self.vocab.get(token, self.vocab[self.special_tokens['unk_token']]) for token in tokens]
            
            # Prepend <s> token and append <\s> token
            token_ids = [self.vocab[self.special_tokens['bos_token']]] + token_ids + [self.vocab[self.special_tokens['eos_token']]]
            
            # Create attention mask
            attention_mask = [1] * len(token_ids)
            
            # Padding or truncating to the max_seq_length
            if len(token_ids) < self.max_seq_length:
                padding_length = self.max_seq_length - len(token_ids)
                token_ids += [self.vocab[self.special_tokens['pad_token']]] * padding_length
                attention_mask += [0] * padding_length
            else:
                token_ids = token_ids[:self.max_seq_length]
                attention_mask = attention_mask[:self.max_seq_length]

            batch_token_ids.append(torch.tensor(token_ids))
            batch_attention_masks.append(torch.tensor(attention_mask))

        return {
            "token_ids": torch.stack(batch_token_ids),
            "attention_mask": torch.stack(batch_attention_masks)
        }

    def decode(self, batch_token_ids, skip_special_tokens=False):
        if batch_token_ids.dim() == 1:
            batch_token_ids = batch_token_ids.unsqueeze(0)

        decoded_texts = []
        for token_ids in batch_token_ids:
            if skip_special_tokens:
                decoded_texts.append(' '.join([self.reverse_vocab[token_id.item()] for token_id in token_ids if token_id.item() not in [self.vocab[val] for val in self.special_tokens.values()]]))
            else:
                decoded_texts.append(' '.join([self.reverse_vocab[token_id.item()] for token_id in token_ids if token_id.item() != self.vocab[self.special_tokens['pad_token']]]))

        return decoded_texts if len(decoded_texts) > 1 else decoded_texts[0]

    
    def save_vocabulary(self, path="vocab.json"):
        with open(path, 'w') as file:
            json.dump(self.vocab, file)

    @property
    def vocab_size(self):
        """Returns the size of the vocabulary."""
        return len(self.vocab)

    @classmethod
    def load_vocabulary(cls, path="vocab.json", max_seq_length=512):
        with open(path, 'r') as file:
            loaded_vocab = json.load(file)
        return cls(list(loaded_vocab.keys()), max_seq_length)    

    
class GlycanTranslationData(Dataset):
    def __init__(self, input_corpus, output_corpus, pad_token_id, eos_token_id):
        self.input_ids = input_corpus["token_ids"]
        self.input_attention_masks = input_corpus["attention_mask"]
        
        self.output_ids = output_corpus["token_ids"]
        self.output_attention_masks = output_corpus["attention_mask"]

        # Set pad_token_id, bos_token_id
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        # Extract the output_ids for the given idx
        output_ids_for_idx = self.output_ids[idx]
    
        # If output_ids_for_idx is a 1D tensor, we need to add an extra batch dimension
        if len(output_ids_for_idx.shape) == 1:
            output_ids_for_idx = output_ids_for_idx.unsqueeze(0)

        # Using shift_tokens_right to create decoder_input_ids
        decoder_input_ids = shift_tokens_right(output_ids_for_idx, self.pad_token_id, self.eos_token_id).squeeze(0)

        # Prepend a value of 1 (indicating attention) to the attention mask 
        # and then remove the last value to match the length of decoder_input_ids.
        decoder_attention_mask = torch.cat([torch.tensor([1]), self.output_attention_masks[idx]])[:-1]
       
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.input_attention_masks[idx],
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": self.output_attention_masks[idx],
            "labels": self.output_ids[idx]
        }
