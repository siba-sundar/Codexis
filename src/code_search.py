

import os
import sys
import re
import torch
import argparse
import pickle
from torch import nn
import numpy as np


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=5000):
        super(PositionalEncoding, self).__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register buffer (won't be updated during backprop)
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        # Add positional encoding to input
        return x + self.pe[:, :x.size(1)]


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        # Linear layers for Q, K, V
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        
        self.out = nn.Linear(d_model, d_model)
        
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        # Linear projections and reshape for multi-head
        q = self.q_linear(q).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(k).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(v).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax to get attention weights
        attention = torch.softmax(scores, dim=-1)
        
        # Apply attention weights to values
        out = torch.matmul(attention, v)
        
        # Reshape and pass through final linear layer
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
        # Self attention with residual connection and layer norm
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed forward with residual connection and layer norm
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
        # Convert input tokens to embeddings
        x = self.embedding(x) * np.sqrt(self.embedding.embedding_dim)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Apply dropout
        x = self.dropout(x)
        
        # Pass through encoder layers
        for layer in self.layers:
            x = layer(x, mask)
        
        # Final layer normalization
        return self.norm(x)


class CodeDocTransformer(nn.Module):
    def __init__(self, code_vocab_size, doc_vocab_size, d_model=256, num_heads=8, 
                 num_layers=3, d_ff=512, max_seq_len=512, dropout=0.1):
        super(CodeDocTransformer, self).__init__()
        
        # Code and docstring encoders
        self.code_encoder = Encoder(code_vocab_size, d_model, num_heads, num_layers, 
                                   d_ff, max_seq_len, dropout)
        self.doc_encoder = Encoder(doc_vocab_size, d_model, num_heads, num_layers, 
                                  d_ff, max_seq_len, dropout)
        
        # Projection for similarity calculation
        self.code_projection = nn.Linear(d_model, d_model)
        self.doc_projection = nn.Linear(d_model, d_model)
        
    def forward(self, code_ids, doc_ids, code_mask=None, doc_mask=None):
        # Encode code and docstring
        code_encoded = self.code_encoder(code_ids, code_mask)
        doc_encoded = self.doc_encoder(doc_ids, doc_mask)
        
        # Get sequence representations (mean pooling)
        code_repr = code_encoded.mean(dim=1)
        doc_repr = doc_encoded.mean(dim=1)
        
        # Project to common space
        code_proj = self.code_projection(code_repr)
        doc_proj = self.doc_projection(doc_repr)
        
        return code_proj, doc_proj
    
    
class SimpleTokenizer:
    
    
    def __init__(self):
        self.vocab = {}  # This will be populated when loaded from pickle
        self.max_len = 512
        
    def encode(self, text, max_length=None):
        """Convert text to token IDs."""
        if max_length is None:
            max_length = self.max_len
            
        # Since we don't know the exact tokenization method used,
        # we'll provide a simple fallback for testing
        if not hasattr(self, 'vocab') or not self.vocab:
            # Fallback tokenization
            tokens = text.lower().split()
            return [i+1 for i, _ in enumerate(tokens[:max_length])]
        
        # If we have a vocabulary, use it
        tokens = text.split()
        ids = []
        for token in tokens[:max_length]:
            if token in self.vocab:
                ids.append(self.vocab[token])
            else:
                ids.append(self.vocab.get('<unk>', 0))
                
        return ids
        
    def decode(self, ids):
      
        if not hasattr(self, 'vocab') or not self.vocab:
            # Can't decode without vocabulary
            return " ".join([f"<token_{id}>" for id in ids])
            
        # Reverse mapping of vocab
        id_to_token = {v: k for k, v in self.vocab.items()}
        
        # Convert IDs to tokens
        tokens = [id_to_token.get(id, '<unk>') for id in ids]
        
        # Join tokens to form text
        return ' '.join(tokens)

def load_model(model_dir, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    
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
        sys.exit(1)


def load_code_corpus(corpus_path):
    
    try:
        with open(corpus_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"Error loading code corpus: {e}")
        sys.exit(1)


def extract_queries_from_file(file_path):
   
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract docstrings (both triple single and double quotes)
        docstrings = re.findall(r'"""(.*?)"""', content, re.DOTALL) + \
                     re.findall(r"'''(.*?)'''", content, re.DOTALL)
        
        # Extract function definitions with one-line comments
        fn_comments = []
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith('def ') and '#' in line:
                comment = line.split('#', 1)[1].strip()
                fn_name = line.split('def ')[1].split('(')[0].strip()
                fn_comments.append(f"{fn_name}: {comment}")
        
        # Extract other comments that might be relevant
        general_comments = []
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith('#') and len(line) > 5:  # Reasonably long comments
                comment = line[1:].strip()
                if not any(kw in comment.lower() for kw in ['todo', 'fixme', 'hack']):
                    general_comments.append(comment)
        
        # Combine all potential queries
        queries = []
        
        # Add docstrings (often most relevant)
        for docstring in docstrings:
            # Process docstring to extract short description
            clean_doc = ' '.join([line.strip() for line in docstring.split('\n')]).strip()
            if clean_doc and len(clean_doc) > 10:  # Non-empty meaningful docstring
                # Get first sentence or limit to reasonable length
                first_part = clean_doc.split('.')[0] if '.' in clean_doc else clean_doc
                if len(first_part) > 200:
                    first_part = first_part[:200] + "..."
                queries.append(first_part)
        
        # Add function comments
        queries.extend(fn_comments)
        
        # Add general comments (if not too many)
        if len(general_comments) < 5:  # Don't include too many general comments
            queries.extend(general_comments)
        else:
            # Pick a few longer, more descriptive comments
            sorted_comments = sorted(general_comments, key=len, reverse=True)
            queries.extend(sorted_comments[:3])
        
        # If no good queries found, fallback to file name or basic code snippets
        if not queries:
            file_name = os.path.basename(file_path)
            base_name = os.path.splitext(file_name)[0].replace('_', ' ').replace('-', ' ')
            queries = [f"Code similar to {base_name}"]
        
        return queries
    
    except Exception as e:
        print(f"Error extracting queries from file: {e}")
        return ["Basic Python code example"]  # Fallback query


def search_code_by_query(model, query, code_corpus, doc_tokenizer, code_tokenizer, top_k=5, device=None):
   
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.eval()
    
    # Simple fallback tokenization if the tokenizers don't work
    try:
        query_tokens = doc_tokenizer.encode(query, max_length=128)
    except AttributeError:
        # Simple fallback tokenization
        query_tokens = [i+1 for i, token in enumerate(query.lower().split()[:128])]
    
    query_tensor = torch.tensor([query_tokens], dtype=torch.long).to(device)
    
    # Encode code corpus with fallback
    code_tensors = []
    for code in code_corpus:
        try:
            code_tokens = code_tokenizer.encode(code, max_length=512)
        except AttributeError:
            # Simple fallback tokenization
            code_tokens = [i+1 for i, token in enumerate(code.lower().split()[:512])]
        code_tensors.append(torch.tensor(code_tokens, dtype=torch.long))
    
    code_batch = torch.nn.utils.rnn.pad_sequence(code_tensors, batch_first=True).to(device)
    
    # Get embeddings
    with torch.no_grad():
        _, query_embed = model(code_batch[:1], query_tensor)  # Dummy code input
        code_embed, _ = model(code_batch, query_tensor[:1].repeat(code_batch.size(0), 1))  # Dummy doc input
    
    # Calculate similarities
    query_embed_norm = query_embed / query_embed.norm(dim=1, keepdim=True)
    code_embed_norm = code_embed / code_embed.norm(dim=1, keepdim=True)
    
    similarities = torch.matmul(query_embed_norm, code_embed_norm.transpose(0, 1)).squeeze(0)
    
    # Get top-k results
    top_indices = similarities.argsort(descending=True)[:top_k]
    
    results = []
    for idx in top_indices:
        results.append({
            'code': code_corpus[idx.item()],
            'similarity': similarities[idx].item()
        })
    
    return results


def print_results(results, query):
    
    print(f"\n=== Query: {query} ===\n")
    
    for i, result in enumerate(results):
        similarity_pct = result['similarity'] * 100
        print(f"\n----- Result {i+1} (Similarity: {similarity_pct:.2f}%) -----")
        
        # Print code with syntax highlighting if available, otherwise plain
        try:
            from pygments import highlight
            from pygments.lexers import PythonLexer
            from pygments.formatters import TerminalFormatter
            
            highlighted_code = highlight(result['code'], PythonLexer(), TerminalFormatter())
            print(highlighted_code)
        except ImportError:
            print(result['code'])
        
        print("-" * 60)


def main():
    parser = argparse.ArgumentParser(description="Search for relevant code snippets based on a Python file")
    parser.add_argument("input_file", help="Path to the Python file to analyze")
    parser.add_argument("--model-dir", default="./code_transformer_export", 
                        help="Directory containing the saved model files")
    parser.add_argument("--corpus", default="./code_corpus.txt", 
                        help="Path to file containing code corpus (one snippet per line)")
    parser.add_argument("--top-k", type=int, default=5, 
                        help="Number of results to display")
    parser.add_argument("--query", help="Use a specific query instead of extracting from file")
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.model_dir):
        print(f"Error: Model directory '{args.model_dir}' not found.")
        sys.exit(1)
    
    if not os.path.exists(args.corpus):
        print(f"Error: Code corpus file '{args.corpus}' not found.")
        sys.exit(1)
    
    if not os.path.exists(args.input_file) and not args.query:
        print(f"Error: Input file '{args.input_file}' not found.")
        sys.exit(1)
    
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
    
    # Get search queries
    if args.query:
        # Use provided query
        queries = [args.query]
    else:
        # Extract queries from file
        print(f"Analyzing Python file: {args.input_file}")
        queries = extract_queries_from_file(args.input_file)
        print(f"Extracted {len(queries)} potential search queries")
    
    # Perform search for each query
    for query in queries:
        results = search_code_by_query(
            model, query, code_corpus, doc_tokenizer, code_tokenizer, top_k=args.top_k, device=device
        )
        print_results(results, query)
        
        # If multiple queries, ask if user wants to continue
        if len(queries) > 1 and query != queries[-1]:
            if input("\nPress Enter to see next query results or 'q' to quit: ").lower() == 'q':
                break


if __name__ == "__main__":
    main()