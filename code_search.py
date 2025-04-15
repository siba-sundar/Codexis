import os
import torch
import pickle
import argparse
import re
from collections import Counter
from torch import nn
import numpy as np


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        
        self.out = nn.Linear(d_model, d_model)
        
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        q = self.q_linear(q).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(k).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(v).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention = torch.softmax(scores, dim=-1)
        
        out = torch.matmul(attention, v)
        
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.out(out)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_len, dropout=0.1):
        super(Encoder, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x, mask=None):
        x = self.embedding(x) * np.sqrt(self.embedding.embedding_dim)
        
        x = self.pos_encoding(x)
        
        x = self.dropout(x)
        
        for layer in self.layers:
            x = layer(x, mask)
        
        return self.norm(x)


class CodeDocTransformer(nn.Module):
    def __init__(self, code_vocab_size, doc_vocab_size, d_model=256, num_heads=8, 
                 num_layers=3, d_ff=512, max_seq_len=512, dropout=0.1):
        super(CodeDocTransformer, self).__init__()
        
        self.code_encoder = Encoder(code_vocab_size, d_model, num_heads, num_layers, 
                                   d_ff, max_seq_len, dropout)
        self.doc_encoder = Encoder(doc_vocab_size, d_model, num_heads, num_layers, 
                                  d_ff, max_seq_len, dropout)
        
        self.code_projection = nn.Linear(d_model, d_model)
        self.doc_projection = nn.Linear(d_model, d_model)
        
    def forward(self, code_ids, doc_ids, code_mask=None, doc_mask=None):
        code_encoded = self.code_encoder(code_ids, code_mask)
        doc_encoded = self.doc_encoder(doc_ids, doc_mask)
        
        code_repr = code_encoded.mean(dim=1)
        doc_repr = doc_encoded.mean(dim=1)
        
        code_proj = self.code_projection(code_repr)
        doc_proj = self.doc_projection(doc_repr)
        
        return code_proj, doc_proj


class SimpleTokenizer:
    def __init__(self, texts=None, vocab_size=10000, min_freq=2):
        self.vocab_size = vocab_size
        self.min_freq = min_freq
        self.word2idx = {'<PAD>': 0, '<UNK>': 1, '< SOS >': 2, '<EOS>': 3}
        self.idx2word = {0: '<PAD>', 1: '<UNK>', 2: '< SOS >', 3: '<EOS>'}
        self.word_freq = Counter()
        self.max_len = 512
        
        if texts:
            self.fit(texts)
    
    def tokenize(self, text):
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        return text.split()
    
    def fit(self, texts):
        for text in texts:
            self.word_freq.update(self.tokenize(text))
        
        words = [word for word, count in self.word_freq.items() 
                if count >= self.min_freq]
        
        words = sorted(words, key=lambda x: self.word_freq[x], reverse=True)[:self.vocab_size - 4]
        
        for i, word in enumerate(words):
            idx = i + 4 
            self.word2idx[word] = idx
            self.idx2word[idx] = word
    
    def encode(self, text, max_length=None):
        if max_length is None:
            max_length = self.max_len
            
        tokens = [self.word2idx.get(word, self.word2idx['<UNK>']) 
                for word in self.tokenize(text)]
        
        # Check for different possible SOS token formats
        sos_token = None
        for possible_sos in ['< SOS >', '<SOS>', '<sos>', '< sos >']:
            if possible_sos in self.word2idx:
                sos_token = self.word2idx[possible_sos]
                break
        
        # If no SOS token found, use UNK
        if sos_token is None:
            sos_token = self.word2idx['<UNK>']
            print("Warning: SOS token not found in vocabulary, using UNK instead")
        
        # Check for different possible EOS token formats
        eos_token = None
        for possible_eos in ['<EOS>', '<eos>', '< EOS >', '< eos >']:
            if possible_eos in self.word2idx:
                eos_token = self.word2idx[possible_eos]
                break
        
        # If no EOS token found, use UNK
        if eos_token is None:
            eos_token = self.word2idx['<UNK>']
            print("Warning: EOS token not found in vocabulary, using UNK instead")
        
        tokens = [sos_token] + tokens + [eos_token]
        
        if max_length:
            if len(tokens) > max_length:
                tokens = tokens[:max_length]
            else:
                tokens += [self.word2idx['<PAD>']] * (max_length - len(tokens))
        
        return tokens
    
    def decode(self, ids):
        return ' '.join([self.idx2word.get(idx, '<UNK>') for idx in ids])


def load_model(model_dir, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    """
    Load a previously saved model and tokenizers
    """
    try:
        # Load model configuration
        with open(os.path.join(model_dir, "model_config.pkl"), 'rb') as f:
            model_config = pickle.load(f)
        
        # Recreate the model
        model = CodeDocTransformer(
            code_vocab_size=model_config['code_vocab_size'],
            doc_vocab_size=model_config['doc_vocab_size'],
            d_model=model_config['d_model'],
            num_heads=model_config['num_heads'],
            num_layers=model_config['num_layers'],
            d_ff=model_config['d_ff'],
            max_seq_len=model_config['max_seq_len'],
            dropout=model_config['dropout']
        ).to(device)
        
        # Load state dict
        model.load_state_dict(torch.load(os.path.join(model_dir, "model_state_dict.pt"), map_location=device))
        
        # Load tokenizers
        with open(os.path.join(model_dir, "code_tokenizer.pkl"), 'rb') as f:
            code_tokenizer = pickle.load(f)
        
        with open(os.path.join(model_dir, "doc_tokenizer.pkl"), 'rb') as f:
            doc_tokenizer = pickle.load(f)
        
        print(f"Model and tokenizers loaded from {model_dir}")
        return model, code_tokenizer, doc_tokenizer
    
    except Exception as e:
        print(f"Error loading model: {e}")
        raise


def load_code_corpus(corpus_path):
    """Load code corpus from a text file"""
    try:
        with open(corpus_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"Error loading code corpus: {e}")
        raise


def search_code_by_query(model, query, code_corpus, doc_tokenizer, code_tokenizer, top_k=5, device=None, batch_size=16):
    """Search for code snippets based on a natural language query using batched processing"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.eval()
    
    
    query_tokens = doc_tokenizer.encode(query, max_length=128)
    query_tensor = torch.tensor([query_tokens], dtype=torch.long).to(device)
    
    
    with torch.no_grad():
       
        dummy_code = torch.zeros((1, 1), dtype=torch.long).to(device)
        _, query_embed = model(dummy_code, query_tensor)
    
    
    query_embed_norm = query_embed / query_embed.norm(dim=1, keepdim=True)
    
    
    similarities = []
    
    for i in range(0, len(code_corpus), batch_size):
        batch_corpus = code_corpus[i:i+batch_size]
        
        # Encode code batch
        code_tensors = []
        for code in batch_corpus:
            code_tokens = code_tokenizer.encode(code, max_length=512)
            code_tensors.append(torch.tensor(code_tokens, dtype=torch.long))
        
        code_batch = torch.nn.utils.rnn.pad_sequence(code_tensors, batch_first=True).to(device)
        
       
        with torch.no_grad():
            # We need a dummy doc input for the model's forward pass
            dummy_doc = query_tensor[:1].repeat(code_batch.size(0), 1)
            code_embed, _ = model(code_batch, dummy_doc)
        
        # Normalize code embeddings
        code_embed_norm = code_embed / code_embed.norm(dim=1, keepdim=True)
        
        # Calculate similarities for this batch
        batch_similarities = torch.matmul(query_embed_norm, code_embed_norm.transpose(0, 1)).squeeze(0)
        similarities.append(batch_similarities)
    
    # Combine all similarities
    all_similarities = torch.cat(similarities)
    
    # Get top-k results
    if top_k > len(all_similarities):
        top_k = len(all_similarities)
    
    top_values, top_indices = all_similarities.topk(top_k)
    
    results = []
    for i in range(top_k):
        idx = top_indices[i].item()
        results.append({
            'code': code_corpus[idx],
            'similarity': top_values[i].item()
        })
    
    return results


def print_results(results, query):
    """Print search results to console"""
    print(f"\n=== Query: {query} ===\n")
    
    for i, result in enumerate(results):
        similarity_pct = result['similarity'] * 100
        print(f"\n----- Result {i+1} (Similarity: {similarity_pct:.2f}%) -----")
        print(result['code'])
        print("-" * 60)


def main():
    parser = argparse.ArgumentParser(description="Search for code snippets based on natural language query")
    parser.add_argument("--model-dir", default="./code_transformer_export", 
                        help="Directory containing the saved model files")
    parser.add_argument("--corpus", default="./code_corpus.txt", 
                        help="Path to file containing code corpus (one snippet per line)")
    parser.add_argument("--top-k", type=int, default=5, 
                        help="Number of results to display")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for processing code corpus")
    parser.add_argument("--query", required=True, help="Natural language query to search for code")
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.model_dir):
        print(f"Error: Model directory '{args.model_dir}' not found.")
        return
    
    if not os.path.exists(args.corpus):
        print(f"Error: Code corpus file '{args.corpus}' not found.")
        return
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model and tokenizers
    print(f"Loading model from {args.model_dir}...")
    model, code_tokenizer, doc_tokenizer = load_model(args.model_dir, device)
    
    # Load code corpus
    print(f"Loading code corpus from {args.corpus}...")
    code_corpus = load_code_corpus(args.corpus)
    print(f"Loaded {len(code_corpus)} code snippets")
    
    # Perform search
    print(f"Searching with batch size {args.batch_size}...")
    results = search_code_by_query(
        model, args.query, code_corpus, doc_tokenizer, code_tokenizer, 
        top_k=args.top_k, device=device, batch_size=args.batch_size
    )
    print_results(results, args.query)


if __name__ == "__main__":
    main()