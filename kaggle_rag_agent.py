# %% [markdown]
# # 🧠 MBTI Personality Detection — RAG + Multi-Agent System
#
# ## Architecture: Agent = LLM + Planning + Tool Use + Loop + Memory
#
# ```
#                 ┌──────────────────────────────────┐
#                 │        Input: User Posts          │
#                 └──────────────┬───────────────────┘
#                                │
#                 ┌──────────────▼───────────────────┐
#                 │    Preprocessing Pipeline         │
#                 │  • Mask MBTI keywords (<type>)    │
#                 │  • Truncate (70 words/post)       │
#                 └──────────────┬───────────────────┘
#                                │
#      ┌─────────────────────────▼──────────────────────────┐
#      │              MBTI Analysis Agent                    │
#      │  ┌───────────────────────────────────────────────┐  │
#      │  │ 🤖 LLM: Qwen2.5-7B-Instruct (4-bit NF4)    │  │
#      │  ├───────────────────────────────────────────────┤  │
#      │  │ 📋 Planning: Decompose → 4 structured steps  │  │
#      │  ├───────────────────────────────────────────────┤  │
#      │  │ 🔧 Tools:                                    │  │
#      │  │   • RAG Retriever (FAISS dual-embedding)     │  │
#      │  │   • Linguistic Analyzer (stat features)      │  │
#      │  │   • MBTI Knowledge Lookup                    │  │
#      │  │   • MBTI Predictor (single-shot, all 4 axes) │  │
#      │  ├───────────────────────────────────────────────┤  │
#      │  │ 🔄 Loop: Think → Act → Observe → Repeat     │  │
#      │  │   + Confidence re-check for uncertain axes   │  │
#      │  ├───────────────────────────────────────────────┤  │
#      │  │ 🧠 Memory:                                   │  │
#      │  │   • Working Memory (current analysis state)  │  │
#      │  │   • Long-term Memory (KB + RAG examples)     │  │
#      │  └───────────────────────────────────────────────┘  │
#      └─────────────────────────────────────────────────────┘
#                                │
#      ┌─────────────────────────▼──────────────────────────┐
#      │     Single-Shot MBTI Prediction (1 LLM call)        │
#      │  ┌─────────────────────────────────────────────┐    │
#      │  │  All context (RAG + features + KB markers)  │    │
#      │  │  → LLM predicts all 4 axes simultaneously   │    │
#      │  │  → I/E + S/N + T/F + J/P + confidence       │    │
#      │  └─────────────────────────────────────────────┘    │
#      │     ↓ if any axis confidence < threshold            │
#      │     → Re-analyze uncertain axes (Loop)              │
#      └────────────────────────┬────────────────────────────┘
#                               │
#      ┌────────────────────────▼───────────────────────────┐
#      │  Reasoning Agent (1 LLM call)                      │
#      │  → Generates human-readable explanation            │
#      └────────────────────────┬───────────────────────────┘
#                               │
#                 ┌─────────────▼────────────────────┐
#                 │  Output:                          │
#                 │  • 4 axis predictions (I/E,S/N,..)│
#                 │  • Final MBTI type (e.g., INTJ)   │
#                 │  • Reasoning explanation           │
#                 └──────────────────────────────────┘
# ```
#
# **Hardware**: Optimized for Kaggle T4 GPU (16GB VRAM)
# - LLM: 4-bit NF4 quantization (~4-5GB VRAM)
# - Sentence Encoder: fp16 (~100MB VRAM)
# - FAISS: CPU-based (zero VRAM)

# %% [markdown]
# ## 📦 Cell 1 — Install Dependencies

# %%
# !pip install -q transformers>=4.40.0 accelerate>=0.28.0 bitsandbytes>=0.43.0
# !pip install -q sentence-transformers faiss-cpu scikit-learn tqdm

# %% [markdown]
# ## 📥 Cell 2 — Imports & GPU Detection

# %%
import os
import re
import gc
import json
import time
import warnings
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_class_weight
import faiss

warnings.filterwarnings('ignore')

NUM_GPUS = torch.cuda.device_count()
DEVICE = torch.device('cuda' if NUM_GPUS > 0 else 'cpu')
if NUM_GPUS > 0:
    GPU_NAME = torch.cuda.get_device_name(0)
    GPU_MEM = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f'🖥  GPU: {GPU_NAME} | VRAM: {GPU_MEM:.1f} GB')
else:
    print('⚠️  No GPU found — CPU mode (will be slow)')
print(f'🔧 Device: {DEVICE}')

# %% [markdown]
# ## ⚙️ Cell 3 — Configuration

# %%
CONFIG = {
    # ── Data paths (auto-detect Kaggle vs local) ──
    'DATA_PATH_KAGGLE_500K': '/kaggle/input/mbti-personality-types-500-dataset/MBTI 500.csv',
    'DATA_PATH_KAGGLE_8K': '/kaggle/input/mbti-type/mbti_1.csv',
    'DATA_PATH_LOCAL': 'dataset/mbti_1.csv',
    'RESULT_DIR': '/kaggle/working/results' if os.path.exists('/kaggle') else 'outputs',

    # ── Preprocessing ──
    'MAX_POSTS': 50,
    'MAX_WORDS': 70,

    # ── LLM (local, 4-bit quantized) ──
    'LLM_MODEL': 'Qwen/Qwen2.5-7B-Instruct',
    'LLM_MAX_NEW_TOKENS': 300,
    'LLM_TEMPERATURE': 0.1,

    # ── Sentence Encoder & RAG ──
    'EMBED_MODEL': 'all-MiniLM-L6-v2',
    'RAG_TOP_K': 5,
    'RAG_FETCH_MULT': 3,  # fetch K*mult candidates, then balance

    # ── Agent ──
    'CONFIDENCE_THRESHOLD': 0.65,
    'MAX_REANALYZE_LOOPS': 1,

    # ── Evaluation ──
    'TEST_SAMPLE': None,  # Set to int (e.g. 200) for quick testing, None for full
}

os.makedirs(CONFIG['RESULT_DIR'], exist_ok=True)

# ── Constants ──
MBTI_TYPES = [
    'infj', 'infp', 'intj', 'intp', 'isfj', 'isfp', 'istj', 'istp',
    'enfj', 'enfp', 'entj', 'entp', 'esfj', 'esfp', 'estj', 'estp'
]
MBTI_EXTRA = [
    'introvert', 'extrovert', 'introverted', 'extroverted',
    'sensing', 'intuition', 'intuitive', 'thinking', 'feeling',
    'judging', 'perceiving', 'perceiver', 'judger'
]
ALL_MASK = MBTI_TYPES + MBTI_EXTRA
MASK_RE = re.compile(r'\b(' + '|'.join(ALL_MASK) + r')\b', re.IGNORECASE)

LABEL_COL = 'type'
TEXT_COL = 'posts'
LABEL_COLS = ['label_ie', 'label_sn', 'label_tf', 'label_jp']
DIM_NAMES = ['I/E', 'S/N', 'T/F', 'J/P']

print('✅ Config ready')

# %% [markdown]
# ## 📂 Cell 4 — Load & Preprocess Dataset

# %%
def load_dataset(cfg):
    """Try multiple paths to find the MBTI dataset."""
    paths = [
        cfg['DATA_PATH_KAGGLE_500K'],
        cfg['DATA_PATH_KAGGLE_8K'],
        cfg['DATA_PATH_LOCAL'],
    ]
    for path in paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            print(f'✅ Loaded: {path}  shape={df.shape}')
            return df
    raise FileNotFoundError('❌ No dataset found. Check paths in CONFIG.')


def map_mbti_to_binary(mbti_str):
    """MBTI string → 4 binary labels: (I=0/E=1, S=0/N=1, T=0/F=1, P=0/J=1)"""
    m = mbti_str.strip().upper()
    return (
        int(m[0] == 'E'),  # I=0, E=1
        int(m[1] == 'N'),  # S=0, N=1
        int(m[2] == 'F'),  # T=0, F=1
        int(m[3] == 'J'),  # P=0, J=1
    )


def preprocess_user_posts(raw_str, max_posts, max_words):
    """Split posts, mask MBTI keywords, truncate each to max_words."""
    posts = raw_str.split('|||')
    result = []
    for p in posts[:max_posts]:
        p = MASK_RE.sub('<type>', p.strip())
        words = p.split()[:max_words]
        p = ' '.join(words)
        if p.strip():
            result.append(p)
    return result


# ── Load ──
df_raw = load_dataset(CONFIG)

# ── Binary labels ──
df_raw[['label_ie', 'label_sn', 'label_tf', 'label_jp']] = (
    df_raw[LABEL_COL].apply(lambda x: pd.Series(map_mbti_to_binary(x)))
)

# ── Preprocessing ──
print('⏳ Preprocessing posts (masking + truncation)...')
df_raw['processed_posts'] = df_raw[TEXT_COL].apply(
    lambda x: preprocess_user_posts(x, CONFIG['MAX_POSTS'], CONFIG['MAX_WORDS'])
)
df_raw['concat_posts'] = df_raw['processed_posts'].apply(lambda x: ' ||| '.join(x))
print(f'✅ Preprocessing done. Rows: {len(df_raw)}')

# ── Stratified split ──
df_raw['combo_label'] = df_raw[LABEL_COLS].astype(str).agg(''.join, axis=1)

train_df, temp_df = train_test_split(
    df_raw, test_size=0.4, stratify=df_raw['combo_label'], random_state=42
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.5, stratify=temp_df['combo_label'], random_state=42
)
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

print(f'📊 Split → Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}')

# ── Class weights ──
class_weights_list = []
for col in LABEL_COLS:
    y = train_df[col].values
    cw = compute_class_weight('balanced', classes=np.unique(y), y=y)
    class_weights_list.append(torch.tensor(cw, dtype=torch.float))
print('✅ Class weights:', [cw.tolist() for cw in class_weights_list])

# %% [markdown]
# ## 🤖 Cell 5 — Local LLM Engine (4-bit Quantized)
#
# Load **Qwen2.5-7B-Instruct** with BitsAndBytes NF4 quantization.
# When 2 GPUs are available, loads a **separate model on each GPU** for parallel inference.

# %%
import threading

print('⏳ Loading LLM with 4-bit quantization...')
t0 = time.time()

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

llm_tokenizer = AutoTokenizer.from_pretrained(
    CONFIG['LLM_MODEL'],
    trust_remote_code=True,
)
if llm_tokenizer.pad_token is None:
    llm_tokenizer.pad_token = llm_tokenizer.eos_token

# ── Load model(s) ──
llm_models = []  # list of (model, lock) tuples

def _load_model_on_device(device_id):
    """Load a 4-bit model pinned to a specific GPU."""
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG['LLM_MODEL'],
        quantization_config=bnb_config,
        device_map={'': device_id},
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    model.eval()
    return model

if NUM_GPUS >= 2:
    print(f'🚀 Loading 2 model instances on 2 GPUs for parallel inference...')
    for gpu_id in range(2):
        model = _load_model_on_device(gpu_id)
        llm_models.append((model, threading.Lock()))
        mem = torch.cuda.memory_allocated(gpu_id) / 1e9
        print(f'   GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)} | VRAM used: {mem:.2f} GB')
else:
    model = _load_model_on_device(0) if NUM_GPUS == 1 else AutoModelForCausalLM.from_pretrained(
        CONFIG['LLM_MODEL'],
        quantization_config=bnb_config,
        device_map='auto',
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    model.eval()
    llm_models.append((model, threading.Lock()))
    if torch.cuda.is_available():
        mem = torch.cuda.memory_allocated() / 1e9
        print(f'   VRAM used: {mem:.2f} GB')

# Keep backward-compatible reference
llm_model = llm_models[0][0]

print(f'✅ {len(llm_models)} LLM instance(s) loaded in {time.time()-t0:.1f}s')


def _llm_generate_on_model(model, prompt, max_new_tokens, temperature):
    """Generate text using a specific model instance."""
    messages = [{'role': 'user', 'content': prompt}]
    text = llm_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = llm_tokenizer(text, return_tensors='pt', truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=max(temperature, 0.01),
            do_sample=temperature > 0,
            top_p=0.9,
            pad_token_id=llm_tokenizer.eos_token_id,
        )

    response = llm_tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    )
    return response.strip()


def llm_generate(prompt: str,
                 max_new_tokens: int = CONFIG['LLM_MAX_NEW_TOKENS'],
                 temperature: float = CONFIG['LLM_TEMPERATURE']) -> str:
    """Generate text from the local LLM (uses GPU 0). Returns the assistant's response."""
    return _llm_generate_on_model(llm_models[0][0], prompt, max_new_tokens, temperature)


def make_llm_generate_fn(gpu_idx: int):
    """Create a thread-safe LLM generate function pinned to a specific GPU."""
    model, lock = llm_models[gpu_idx]
    def _generate(prompt: str,
                  max_new_tokens: int = CONFIG['LLM_MAX_NEW_TOKENS'],
                  temperature: float = CONFIG['LLM_TEMPERATURE']) -> str:
        with lock:
            return _llm_generate_on_model(model, prompt, max_new_tokens, temperature)
    return _generate


# Quick sanity check
_test = llm_generate('Say hello in one word.')
print(f'   LLM test: "{_test[:80]}"')

# %% [markdown]
# ## 📚 Cell 6 — MBTI Knowledge Base (Long-Term Memory)
#
# Hard-coded expert knowledge about each MBTI axis — linguistic and
# behavioral markers. This serves as the agent's **long-term memory**.

# %%
MBTI_KNOWLEDGE_BASE = {
    'I/E': {
        'axis_name': 'Introversion (I) vs Extraversion (E)',
        'description': 'Energy direction: inward reflection vs outward engagement.',
        'markers': {
            0: {  # Introvert
                'label': 'Introvert (I)',
                'linguistic': [
                    'More self-referential pronouns (I, me, my, myself)',
                    'Longer, more complex sentence structures',
                    'More hedging language (perhaps, maybe, might, seems)',
                    'Fewer social/group references',
                    'More reflective and abstract topics',
                    'Deeper analysis of fewer topics',
                    'Lower posting frequency but higher content density',
                ],
            },
            1: {  # Extravert
                'label': 'Extravert (E)',
                'linguistic': [
                    'More social/group pronouns (we, us, everyone, people)',
                    'Shorter, more conversational sentences',
                    'More exclamation marks and emphatic language',
                    'References to social activities and events',
                    'Higher emotional expressiveness',
                    'More direct address (you, your)',
                    'Higher posting frequency, broader topic range',
                ],
            },
        },
    },
    'S/N': {
        'axis_name': 'Sensing (S) vs Intuition (N)',
        'description': 'Information processing: concrete details vs abstract patterns.',
        'markers': {
            0: {  # Sensing
                'label': 'Sensing (S)',
                'linguistic': [
                    'Concrete, specific language with sensory details',
                    'References to present experiences and facts',
                    'Practical, step-by-step descriptions',
                    'More common/frequent vocabulary',
                    'Focus on what IS rather than what COULD BE',
                    'Sequential narrative structure',
                    'Fewer metaphors and analogies',
                ],
            },
            1: {  # Intuition
                'label': 'Intuition (N)',
                'linguistic': [
                    'Abstract, conceptual language',
                    'Future-oriented and hypothetical phrasing',
                    'Rich use of metaphors, analogies, and symbolism',
                    'More complex/unusual vocabulary',
                    'Discussion of patterns, theories, possibilities',
                    'Non-linear, associative writing style',
                    'References to meaning, purpose, and big-picture ideas',
                ],
            },
        },
    },
    'T/F': {
        'axis_name': 'Thinking (T) vs Feeling (F)',
        'description': 'Decision-making: logic/analysis vs values/harmony.',
        'markers': {
            0: {  # Thinking
                'label': 'Thinking (T)',
                'linguistic': [
                    'Analytical and logical reasoning language',
                    'Cause-effect structures (because, therefore, thus)',
                    'Fewer emotional words, more neutral tone',
                    'Critique and debate-oriented language',
                    'Focus on systems, mechanisms, efficiency',
                    'Impersonal constructions',
                    'Data and evidence references',
                ],
            },
            1: {  # Feeling
                'label': 'Feeling (F)',
                'linguistic': [
                    'Emotion words (feel, love, happy, sad, care)',
                    'Value-laden language (important, meaningful, beautiful)',
                    'Empathetic expressions and social harmony focus',
                    'Personal anecdotes and relationship references',
                    'Supportive and encouraging tone',
                    'Use of emotional intensifiers (really, so much, truly)',
                    'Focus on people and interpersonal dynamics',
                ],
            },
        },
    },
    'J/P': {
        'axis_name': 'Judging (J) vs Perceiving (P)',
        'description': 'Lifestyle: structured/planned vs flexible/spontaneous.',
        'markers': {
            0: {  # Perceiving
                'label': 'Perceiving (P)',
                'linguistic': [
                    'Open-ended and exploratory language',
                    'More hedging (maybe, could be, what if)',
                    'Spontaneous topic shifts and tangents',
                    'Flexible and adaptable tone',
                    'Fewer definitive statements',
                    'More questions and possibilities',
                    'Less structured, stream-of-consciousness writing',
                ],
            },
            1: {  # Judging
                'label': 'Judging (J)',
                'linguistic': [
                    'Definitive, decisive language (must, should, will)',
                    'Structured, organized writing with clear conclusions',
                    'Goal-oriented and outcome-focused',
                    'More planning language (schedule, plan, organize)',
                    'Stronger opinions stated with certainty',
                    'List-making and categorization tendencies',
                    'Closure-seeking language patterns',
                ],
            },
        },
    },
}

print('✅ MBTI Knowledge Base loaded (4 axes, expert markers)')

# %% [markdown]
# ## 🗄️ Cell 7 — Sentence Encoder + FAISS RAG Index
#
# Build a dual-embedding FAISS index from training data:
# - **Raw text embedding** (post content)
# - **Feature signature embedding** (linguistic feature summary)

# %%
print('⏳ Loading sentence encoder...')
sent_encoder = SentenceTransformer(CONFIG['EMBED_MODEL'], device=str(DEVICE))
EMBED_DIM = sent_encoder.get_sentence_embedding_dimension()
print(f'✅ Sentence encoder ready (dim={EMBED_DIM})')


def compute_linguistic_signature(posts_list: List[str]) -> str:
    """Compute a textual summary of linguistic features for embedding."""
    all_text = ' '.join(posts_list)
    words = all_text.split()
    sentences = re.split(r'[.!?]+', all_text)
    sentences = [s.strip() for s in sentences if s.strip()]

    n_words = max(len(words), 1)
    n_sents = max(len(sentences), 1)

    avg_word_len = np.mean([len(w) for w in words]) if words else 0
    avg_sent_len = n_words / n_sents
    vocab_richness = len(set(w.lower() for w in words)) / n_words
    question_ratio = all_text.count('?') / n_sents
    exclaim_ratio = all_text.count('!') / n_sents

    # Pronoun patterns
    first_person = len(re.findall(r'\b(i|me|my|myself|mine)\b', all_text, re.I)) / n_words
    social_words = len(re.findall(r'\b(we|us|our|everyone|people|together)\b', all_text, re.I)) / n_words
    emotion_words = len(re.findall(r'\b(feel|love|happy|sad|care|beautiful|amazing|wonderful|hate|angry)\b', all_text, re.I)) / n_words
    think_words = len(re.findall(r'\b(think|logic|reason|analyze|system|because|therefore|evidence)\b', all_text, re.I)) / n_words
    hedge_words = len(re.findall(r'\b(maybe|perhaps|possibly|might|could|seems)\b', all_text, re.I)) / n_words
    certainty_words = len(re.findall(r'\b(definitely|always|never|must|certainly|absolutely)\b', all_text, re.I)) / n_words
    abstract_words = len(re.findall(r'\b(concept|theory|pattern|meaning|possibility|idea|philosophy|universe)\b', all_text, re.I)) / n_words

    sig = (
        f'word_len={avg_word_len:.1f} sent_len={avg_sent_len:.1f} '
        f'vocab={vocab_richness:.2f} questions={question_ratio:.2f} '
        f'exclaim={exclaim_ratio:.2f} first_person={first_person:.3f} '
        f'social={social_words:.3f} emotion={emotion_words:.3f} '
        f'thinking={think_words:.3f} hedge={hedge_words:.3f} '
        f'certainty={certainty_words:.3f} abstract={abstract_words:.3f}'
    )
    return sig


def encode_texts(texts, encoder, batch_size=64, desc='Encoding'):
    all_embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc=desc):
        batch = texts[i:i + batch_size]
        embs = encoder.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        all_embs.append(embs)
    return np.vstack(all_embs).astype('float32')


# ── Compute signatures for train set ──
print('⏳ Computing linguistic signatures for train set...')
train_df['ling_signature'] = train_df['processed_posts'].apply(compute_linguistic_signature)

# ── Dual embedding ──
print('⏳ Encoding raw texts...')
raw_texts = train_df['concat_posts'].tolist()
sig_texts = train_df['ling_signature'].tolist()

raw_embs = encode_texts(raw_texts, sent_encoder, desc='Raw text')
sig_embs = encode_texts(sig_texts, sent_encoder, desc='Signature')

# ── Fuse and normalize ──
dual_embs = np.hstack([raw_embs, sig_embs]).astype('float32')
faiss.normalize_L2(dual_embs)
DUAL_DIM = dual_embs.shape[1]

# ── Build FAISS index ──
faiss_index = faiss.IndexFlatIP(DUAL_DIM)
faiss_index.add(dual_embs)
print(f'✅ FAISS index built | Vectors: {faiss_index.ntotal} | Dim: {DUAL_DIM}')

# ── RAG metadata ──
rag_metadata = train_df[LABEL_COLS + ['concat_posts', 'ling_signature', 'type']].copy()
rag_metadata = rag_metadata.reset_index(drop=True)
print('✅ RAG metadata stored')

# %% [markdown]
# ## 🔧 Cell 8 — Agent Tools
#
# Four tools that the agent can invoke during its reasoning loop:
# 1. **RAG Retriever** — find similar profiles via FAISS
# 2. **Linguistic Analyzer** — extract quantitative features
# 3. **Knowledge Lookup** — query MBTI theory
# 4. **MBTI Predictor** — single-shot LLM prediction of all 4 axes simultaneously

# %%
# ═══════════════════════════════════════════════════════════
# TOOL 1: RAG Retriever
# ═══════════════════════════════════════════════════════════

class RAGRetrieverTool:
    """Retrieve similar user profiles from the FAISS knowledge base."""

    name = 'retrieve_similar_profiles'
    description = 'Find similar user profiles from the training database using dual-embedding cosine similarity with balanced class selection.'

    def __init__(self, index, metadata_df, encoder, embed_dim, class_weights, top_k=5):
        self.index = index
        self.meta = metadata_df
        self.encoder = encoder
        self.embed_dim = embed_dim
        self.cw = class_weights
        self.top_k = top_k

    def run(self, raw_text: str, signature: str, top_k: int = None) -> List[dict]:
        k = top_k or self.top_k
        raw_emb = self.encoder.encode([raw_text], convert_to_numpy=True).astype('float32')
        sig_emb = self.encoder.encode([signature], convert_to_numpy=True).astype('float32')
        q = np.hstack([raw_emb, sig_emb]).astype('float32')
        faiss.normalize_L2(q)

        k_fetch = min(k * CONFIG['RAG_FETCH_MULT'], self.index.ntotal)
        scores, idxs = self.index.search(q, k_fetch)
        idxs = idxs[0]
        scores = scores[0]

        candidates = self.meta.iloc[idxs].copy()
        candidates['_sim_score'] = scores

        # Balanced selection: ensure minority class coverage
        selected_idxs = set()
        for col_idx, col in enumerate(LABEL_COLS):
            cw = self.cw[col_idx]
            minority_cls = int(cw.argmax().item())
            minority_rows = candidates[candidates[col] == minority_cls]
            if len(minority_rows) > 0:
                selected_idxs.add(minority_rows.index[0])

        for idx in candidates.index:
            if len(selected_idxs) >= k:
                break
            selected_idxs.add(idx)

        result = self.meta.loc[list(selected_idxs)].head(k)
        return result.to_dict(orient='records')


# ═══════════════════════════════════════════════════════════
# TOOL 2: Linguistic Analyzer
# ═══════════════════════════════════════════════════════════

class LinguisticAnalyzerTool:
    """Extract quantitative linguistic features from user posts."""

    name = 'analyze_linguistic_features'
    description = 'Compute statistical linguistic features: word/sentence length, pronoun ratios, emotion/logic word frequencies, vocabulary richness, etc.'

    def run(self, posts_list: List[str]) -> dict:
        all_text = ' '.join(posts_list)
        words = all_text.split()
        sentences = re.split(r'[.!?]+', all_text)
        sentences = [s.strip() for s in sentences if s.strip()]

        n_words = max(len(words), 1)
        n_sents = max(len(sentences), 1)
        n_posts = max(len(posts_list), 1)

        word_lengths = [len(w) for w in words]

        features = {
            'n_posts': n_posts,
            'n_words': n_words,
            'n_sentences': n_sents,
            'avg_word_length': round(np.mean(word_lengths), 2) if word_lengths else 0,
            'avg_sentence_length': round(n_words / n_sents, 2),
            'avg_post_length_words': round(n_words / n_posts, 2),
            'vocabulary_richness': round(len(set(w.lower() for w in words)) / n_words, 3),
            'question_ratio': round(all_text.count('?') / n_sents, 3),
            'exclamation_ratio': round(all_text.count('!') / n_sents, 3),
            'url_count': len(re.findall(r'https?://\S+', all_text)),
            'ellipsis_count': all_text.count('...'),

            # Pronouns
            'first_person_ratio': round(
                len(re.findall(r'\b(i|me|my|myself|mine)\b', all_text, re.I)) / n_words, 4),
            'social_pronoun_ratio': round(
                len(re.findall(r'\b(we|us|our|everyone|people|together|they|them)\b', all_text, re.I)) / n_words, 4),

            # Cognitive style
            'emotion_word_ratio': round(
                len(re.findall(r'\b(feel|feeling|love|happy|sad|care|beautiful|amazing|wonderful|hate|angry|joy|fear|hope|wish)\b', all_text, re.I)) / n_words, 4),
            'thinking_word_ratio': round(
                len(re.findall(r'\b(think|thought|logic|logical|reason|analyze|analysis|system|because|therefore|evidence|theory|prove|fact)\b', all_text, re.I)) / n_words, 4),
            'hedge_word_ratio': round(
                len(re.findall(r'\b(maybe|perhaps|possibly|might|could|seems|tend|somewhat|quite)\b', all_text, re.I)) / n_words, 4),
            'certainty_word_ratio': round(
                len(re.findall(r'\b(definitely|always|never|must|certainly|absolutely|clearly|obviously|sure)\b', all_text, re.I)) / n_words, 4),
            'abstract_word_ratio': round(
                len(re.findall(r'\b(concept|theory|pattern|meaning|possibility|idea|philosophy|universe|imagine|wonder|abstract)\b', all_text, re.I)) / n_words, 4),
        }

        # Derived signals
        features['ie_signal'] = round(features['social_pronoun_ratio'] - features['first_person_ratio'], 4)
        features['tf_signal'] = round(features['emotion_word_ratio'] - features['thinking_word_ratio'], 4)
        features['jp_signal'] = round(features['certainty_word_ratio'] - features['hedge_word_ratio'], 4)
        features['sn_signal'] = round(features['abstract_word_ratio'] + features['vocabulary_richness'] - 0.5, 4)

        return features


# ═══════════════════════════════════════════════════════════
# TOOL 3: Knowledge Lookup
# ═══════════════════════════════════════════════════════════

class KnowledgeLookupTool:
    """Query the MBTI knowledge base for axis-specific markers."""

    name = 'lookup_mbti_knowledge'
    description = 'Retrieve expert knowledge about a specific MBTI axis including linguistic and behavioral markers for each pole.'

    def __init__(self, kb: dict):
        self.kb = kb

    def run(self, axis: str) -> dict:
        if axis not in self.kb:
            return {'error': f'Unknown axis: {axis}'}
        return self.kb[axis]


# ═══════════════════════════════════════════════════════════
# TOOL 4: Single-Shot MBTI Predictor (all 4 axes at once)
# ═══════════════════════════════════════════════════════════

MBTI_PREDICT_PROMPT = '''\
You are a psycholinguistic expert analyzing social media posts to determine MBTI personality type.
You must predict ALL 4 MBTI axes simultaneously based on the evidence below.

=== MBTI AXES & LINGUISTIC MARKERS ===

1. I/E — Introversion vs Extraversion (energy direction)
   Introvert (I=0): self-referential pronouns, longer complex sentences, hedging language, reflective topics, fewer social references
   Extravert (E=1): social/group pronouns, shorter conversational sentences, exclamation marks, social activities, direct address

2. S/N — Sensing vs Intuition (information processing)
   Sensing (S=0): concrete specific language, present-focused, practical step-by-step, common vocabulary, sequential narrative
   Intuition (N=1): abstract conceptual language, future-oriented, metaphors/analogies, complex vocabulary, patterns/theories/possibilities

3. T/F — Thinking vs Feeling (decision-making)
   Thinking (T=0): analytical logical language, cause-effect structures, neutral tone, critique/debate, systems/efficiency focus
   Feeling (F=1): emotion words, value-laden language, empathy, personal anecdotes, supportive tone, relationship references

4. J/P — Judging vs Perceiving (lifestyle)
   Perceiving (P=0): open-ended exploratory language, hedging, spontaneous topic shifts, flexible tone, more questions
   Judging (J=1): definitive decisive language, structured organized writing, goal-oriented, planning language, strong opinions

=== USER'S LINGUISTIC FEATURES ===
{features}

=== SIMILAR PROFILES FROM DATABASE ===
{examples}

=== USER'S POSTS (sample) ===
{posts}

Analyze the writing style holistically. Consider how the 4 axes interact (e.g., IN types often discuss abstract theories, ET types are socially expressive).

Output ONLY a valid JSON object with exactly these keys:
{{"IE": 0_or_1, "IE_conf": 0.0_to_1.0, "IE_evidence": "brief_reason",
  "SN": 0_or_1, "SN_conf": 0.0_to_1.0, "SN_evidence": "brief_reason",
  "TF": 0_or_1, "TF_conf": 0.0_to_1.0, "TF_evidence": "brief_reason",
  "JP": 0_or_1, "JP_conf": 0.0_to_1.0, "JP_evidence": "brief_reason"}}'''


class MBTIPredictorTool:
    """Use LLM to predict all 4 MBTI axes in a single call."""

    name = 'predict_mbti'
    description = 'Use the LLM to analyze posts and predict all 4 MBTI axes (I/E, S/N, T/F, J/P) simultaneously, leveraging cross-axis correlations for better accuracy.'

    def __init__(self, llm_fn, kb: dict):
        self.llm_fn = llm_fn
        self.kb = kb

    def run(self, posts_text: str, features: dict,
            examples: List[dict], reanalyze_axes: List[str] = None) -> dict:
        """
        Predict all 4 axes (or specific axes if reanalyze_axes is set).
        Returns dict with per-axis predictions.
        """
        # Build examples string with MBTI types
        ex_strs = []
        for i, ex in enumerate(examples[:5]):
            mbti_type = ex.get('type', '????')
            snippet = ex.get('concat_posts', '')[:150]
            ex_strs.append(f'  Profile {i+1}: {mbti_type} | Posts: "{snippet}..."')
        examples_str = '\n'.join(ex_strs) if ex_strs else '  (none available)'

        # Build features string
        feat_items = [f'  {k}: {v}' for k, v in features.items() if not k.startswith('n_')]
        features_str = '\n'.join(feat_items[:15])

        prompt = MBTI_PREDICT_PROMPT.format(
            features=features_str,
            examples=examples_str,
            posts=posts_text[:800],
        )

        # If re-analyzing, add a hint to focus on specific axes
        if reanalyze_axes:
            axes_str = ', '.join(reanalyze_axes)
            prompt += f'\n\nIMPORTANT: Pay extra attention to axes: {axes_str}. Analyze more carefully with additional context provided.'

        raw = self.llm_fn(prompt, max_new_tokens=200, temperature=0.1)
        return self._parse_response(raw)

    def _parse_response(self, raw: str) -> dict:
        """Parse LLM JSON response into per-axis predictions."""
        axis_map = {'IE': 'I/E', 'SN': 'S/N', 'TF': 'T/F', 'JP': 'J/P'}
        result = {}

        # Try JSON parsing
        try:
            match = re.search(r'\{[^{}]+\}', raw, re.DOTALL)
            if match:
                parsed = json.loads(match.group())
                for short_key, axis in axis_map.items():
                    pred = int(parsed.get(short_key, 0))
                    conf = float(parsed.get(f'{short_key}_conf', 0.5))
                    evid = str(parsed.get(f'{short_key}_evidence', ''))
                    result[axis] = {
                        'axis': axis,
                        'prediction': min(max(pred, 0), 1),
                        'confidence': min(max(conf, 0.0), 1.0),
                        'evidence': evid,
                    }
                return result
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback: regex extraction
        for short_key, axis in axis_map.items():
            pred = 0
            conf = 0.5
            pred_match = re.search(rf'"{short_key}"\s*:\s*(\d)', raw)
            conf_match = re.search(rf'"{short_key}_conf"\s*:\s*([\d.]+)', raw)
            if pred_match:
                pred = int(pred_match.group(1))
            if conf_match:
                conf = float(conf_match.group(1))
            result[axis] = {
                'axis': axis,
                'prediction': min(max(pred, 0), 1),
                'confidence': min(max(conf, 0.0), 1.0),
                'evidence': 'parsed from partial response',
            }

        return result


# ── Instantiate all tools ──
tool_retriever = RAGRetrieverTool(
    index=faiss_index, metadata_df=rag_metadata, encoder=sent_encoder,
    embed_dim=EMBED_DIM, class_weights=class_weights_list, top_k=CONFIG['RAG_TOP_K'],
)
tool_analyzer = LinguisticAnalyzerTool()
tool_knowledge = KnowledgeLookupTool(MBTI_KNOWLEDGE_BASE)
tool_predictor = MBTIPredictorTool(llm_generate, MBTI_KNOWLEDGE_BASE)

print('✅ All 4 Agent Tools instantiated')

# %% [markdown]
# ## 🧠 Cell 9 — Agent Core: Working Memory + ReAct Loop
#
# The **MBTIAnalysisAgent** implements the full agent architecture:
# - **Planning**: Decomposes the task into 4 structured steps
# - **Tool Use**: Calls tools based on the current plan step
# - **Loop**: Think → Act → Observe → Update Memory → (optionally re-analyze uncertain axes)
# - **Memory**: Working memory tracks all observations and decisions
# - **Single-shot prediction**: 1 LLM call predicts all 4 axes simultaneously (3x faster, captures cross-axis correlations)

# %%
@dataclass
class WorkingMemory:
    """Agent's working memory for a single user analysis."""
    user_posts: List[str] = field(default_factory=list)
    concat_text: str = ''
    linguistic_features: dict = field(default_factory=dict)
    ling_signature: str = ''
    similar_profiles: List[dict] = field(default_factory=list)
    axis_knowledge: Dict[str, dict] = field(default_factory=dict)
    axis_predictions: Dict[str, dict] = field(default_factory=dict)
    reasoning_trace: List[str] = field(default_factory=list)
    reanalysis_count: int = 0

    def add_trace(self, step: str, thought: str, observation: str):
        self.reasoning_trace.append(
            f'[Step: {step}] Thought: {thought} | Observation: {observation[:200]}'
        )


class MBTIAnalysisAgent:
    """
    Agent = LLM + Planning + Tool Use + Loop + Memory

    Implements a structured ReAct loop for MBTI personality prediction
    from social media posts.
    """

    def __init__(self, tools: dict, llm_fn, confidence_threshold: float = 0.65,
                 max_reanalyze: int = 1):
        self.tools = tools
        self.llm_fn = llm_fn
        self.confidence_threshold = confidence_threshold
        self.max_reanalyze = max_reanalyze

    # ──────────────────────────────────────────────
    # PLANNING: Decompose the task
    # ──────────────────────────────────────────────
    def plan(self) -> List[dict]:
        """Create an ordered plan of analysis steps."""
        return [
            {
                'step': 1,
                'name': 'analyze_linguistics',
                'tool': 'analyze_linguistic_features',
                'thought': 'First, I need to extract quantitative linguistic features from the user posts to identify measurable patterns.',
            },
            {
                'step': 2,
                'name': 'retrieve_similar',
                'tool': 'retrieve_similar_profiles',
                'thought': 'Next, I should find similar user profiles from the knowledge base to use as reference points.',
            },
            {
                'step': 3,
                'name': 'predict_all_axes',
                'tool': 'predict_mbti',
                'thought': 'Now I have enough context (features + examples + KB markers baked into prompt) to predict all 4 MBTI axes in a single holistic analysis.',
            },
            {
                'step': 4,
                'name': 'confidence_check',
                'tool': None,
                'thought': 'Finally, I should check prediction confidence and re-analyze any uncertain axes with more RAG context.',
            },
        ]

    # ──────────────────────────────────────────────
    # TOOL USE: Execute a specific tool
    # ──────────────────────────────────────────────
    def use_tool(self, tool_name: str, **kwargs) -> Any:
        """Invoke a tool by name and return its output."""
        tool = self.tools.get(tool_name)
        if tool is None:
            return {'error': f'Tool {tool_name} not found'}
        return tool.run(**kwargs)

    # ──────────────────────────────────────────────
    # MAIN LOOP: Think → Act → Observe → Repeat
    # ──────────────────────────────────────────────
    def run(self, user_posts: List[str], skip_reasoning: bool = False) -> dict:
        """
        Execute the full agent loop for one user.

        Args:
            user_posts: List of preprocessed posts
            skip_reasoning: If True, skip the reasoning LLM call (faster batch mode)

        Returns: {
            'axes': {'I/E': 0/1, 'S/N': 0/1, 'T/F': 0/1, 'J/P': 0/1},
            'mbti_type': 'XXXX',
            'confidence': {'I/E': float, ...},
            'reasoning': str,
            'trace': [str, ...]
        }
        """
        # ── Initialize Working Memory ──
        memory = WorkingMemory()
        memory.user_posts = user_posts
        memory.concat_text = ' ||| '.join(user_posts)

        # ── Create Plan ──
        plan = self.plan()

        # ── Execute Plan (Loop) ──
        for step_info in plan:
            step_name = step_info['name']
            thought = step_info['thought']

            if step_name == 'analyze_linguistics':
                observation = self.use_tool(
                    'analyze_linguistic_features',
                    posts_list=memory.user_posts
                )
                memory.linguistic_features = observation
                memory.ling_signature = compute_linguistic_signature(memory.user_posts)
                memory.add_trace(step_name, thought, str(observation))

            elif step_name == 'retrieve_similar':
                observation = self.use_tool(
                    'retrieve_similar_profiles',
                    raw_text=memory.concat_text,
                    signature=memory.ling_signature,
                )
                memory.similar_profiles = observation
                memory.add_trace(step_name, thought,
                                 f'Retrieved {len(observation)} similar profiles')

            elif step_name == 'predict_all_axes':
                # Single LLM call → predict all 4 axes simultaneously
                all_preds = self.use_tool(
                    'predict_mbti',
                    posts_text=memory.concat_text,
                    features=memory.linguistic_features,
                    examples=memory.similar_profiles,
                )
                memory.axis_predictions = all_preds
                pred_summary = ', '.join(
                    f'{ax}={p["prediction"]}(conf={p["confidence"]:.2f})'
                    for ax, p in all_preds.items()
                )
                memory.add_trace(step_name, thought, pred_summary)

            elif step_name == 'confidence_check':
                self._confidence_recheck(memory)

        # ── Synthesize Final Output ──
        return self._synthesize(memory, skip_reasoning=skip_reasoning)

    # ──────────────────────────────────────────────
    # LOOP EXTENSION: Confidence re-check
    # ──────────────────────────────────────────────
    def _confidence_recheck(self, memory: WorkingMemory):
        """
        Re-analyze axes with low confidence using refined context.
        This is the dynamic part of the loop — only triggers when needed.
        Uses a single LLM call with extra RAG examples, focusing on uncertain axes.
        """
        if memory.reanalysis_count >= self.max_reanalyze:
            return

        uncertain_axes = [
            axis for axis, pred in memory.axis_predictions.items()
            if pred['confidence'] < self.confidence_threshold
        ]

        if not uncertain_axes:
            memory.add_trace('confidence_check',
                             'All axes above confidence threshold',
                             'No re-analysis needed')
            return

        memory.reanalysis_count += 1
        memory.add_trace('confidence_check',
                         f'Axes below threshold: {uncertain_axes}',
                         f'Re-analyzing with more context (attempt {memory.reanalysis_count})')

        # Re-retrieve with more examples for better context
        more_examples = self.use_tool(
            'retrieve_similar_profiles',
            raw_text=memory.concat_text,
            signature=memory.ling_signature,
            top_k=CONFIG['RAG_TOP_K'] * 2,
        )

        # Single re-prediction call, hinting to focus on uncertain axes
        new_preds = self.use_tool(
            'predict_mbti',
            posts_text=memory.concat_text,
            features=memory.linguistic_features,
            examples=more_examples,
            reanalyze_axes=uncertain_axes,
        )

        for axis in uncertain_axes:
            if axis in new_preds:
                new_pred = new_preds[axis]
                # Only update if new confidence is higher
                if new_pred['confidence'] > memory.axis_predictions[axis]['confidence']:
                    memory.axis_predictions[axis] = new_pred
                    memory.add_trace(
                        f'reanalyze_{axis}',
                        f'Re-analyzed {axis} with {len(more_examples)} examples',
                        f'Updated: {axis}={new_pred["prediction"]} (conf={new_pred["confidence"]:.2f})'
                    )

    # ──────────────────────────────────────────────
    # SYNTHESIS: Combine predictions + Generate reasoning
    # ──────────────────────────────────────────────
    def _synthesize(self, memory: WorkingMemory, skip_reasoning: bool = False) -> dict:
        """Combine axis predictions into final MBTI type with reasoning."""
        axes = {}
        confidences = {}
        evidences = {}

        for axis in DIM_NAMES:
            pred = memory.axis_predictions.get(axis, {})
            axes[axis] = pred.get('prediction', 0)
            confidences[axis] = pred.get('confidence', 0.5)
            evidences[axis] = pred.get('evidence', '')

        # Build MBTI type string
        mbti_map = {
            'I/E': {0: 'I', 1: 'E'},
            'S/N': {0: 'S', 1: 'N'},
            'T/F': {0: 'T', 1: 'F'},
            'J/P': {0: 'P', 1: 'J'},
        }
        mbti_type = ''.join(mbti_map[axis][axes[axis]] for axis in DIM_NAMES)

        # Generate reasoning from synthesis agent (skip in fast batch mode)
        if skip_reasoning:
            reasoning = '; '.join(
                f'{ax}: {evidences[ax]}' for ax in DIM_NAMES if evidences[ax]
            ) or 'batch mode — reasoning skipped'
        else:
            reasoning = self._generate_reasoning(
                axes, confidences, evidences, memory.linguistic_features, mbti_type
            )

        return {
            'axes': axes,
            'mbti_type': mbti_type,
            'confidence': confidences,
            'reasoning': reasoning,
            'trace': memory.reasoning_trace,
        }

    def _generate_reasoning(self, axes, confidences, evidences,
                            features, mbti_type) -> str:
        """Use LLM to generate a human-readable reasoning summary."""
        axis_summary = []
        for axis in DIM_NAMES:
            pred_label = {
                'I/E': {0: 'Introvert', 1: 'Extravert'},
                'S/N': {0: 'Sensing', 1: 'Intuition'},
                'T/F': {0: 'Thinking', 1: 'Feeling'},
                'J/P': {0: 'Perceiving', 1: 'Judging'},
            }[axis][axes[axis]]
            axis_summary.append(
                f'{axis}: {pred_label} (conf={confidences[axis]:.2f}) — {evidences[axis]}'
            )

        key_feats = {
            'first_person_ratio': features.get('first_person_ratio', 0),
            'social_pronoun_ratio': features.get('social_pronoun_ratio', 0),
            'emotion_word_ratio': features.get('emotion_word_ratio', 0),
            'thinking_word_ratio': features.get('thinking_word_ratio', 0),
            'vocabulary_richness': features.get('vocabulary_richness', 0),
            'question_ratio': features.get('question_ratio', 0),
        }

        prompt = f'''You are an MBTI psycholinguistic expert. Summarize the MBTI analysis in 3-4 sentences.

Predicted type: {mbti_type}
Axis predictions:
{chr(10).join(axis_summary)}

Key linguistic features: {key_feats}

Write a concise, professional explanation of why this person is likely {mbti_type}.
Output ONLY the reasoning text, no JSON.'''

        reasoning = self.llm_fn(prompt, max_new_tokens=200, temperature=0.3)
        return reasoning


print('✅ MBTIAnalysisAgent defined (LLM + Planning + Tool Use + Loop + Memory)')

# %% [markdown]
# ## 🤝 Cell 10 — Agent Orchestrator (Dual-GPU Parallel)
#
# The orchestrator runs the MBTIAnalysisAgent on batches of users.
# **Optimizations for 2×T4 GPUs:**
# - Loads separate LLM on each GPU → 2× throughput via threading
# - `skip_reasoning=True` in batch mode → saves 1 LLM call/sample (~40% faster)
# - Combined: ~3-4× speedup vs single-GPU sequential

# %%
from concurrent.futures import ThreadPoolExecutor, as_completed


class MultiAgentOrchestrator:
    """
    Orchestrates MBTI prediction for multiple users.
    Supports dual-GPU parallel inference when 2 GPUs are available.
    """

    def __init__(self, agents: List[MBTIAnalysisAgent]):
        """
        Args:
            agents: List of MBTIAnalysisAgent instances (one per GPU).
                    For single GPU, pass a list with one agent.
        """
        self.agents = agents
        self.n_workers = len(agents)

    def predict_single(self, user_posts: List[str], agent_idx: int = 0) -> dict:
        """Predict MBTI for a single user using specified agent."""
        return self.agents[agent_idx].run(user_posts)

    def predict_batch(self, df: pd.DataFrame,
                      sample_n: Optional[int] = None,
                      verbose: bool = True,
                      skip_reasoning: bool = True) -> Tuple[np.ndarray, List[dict]]:
        """
        Predict MBTI for a batch of users with parallel GPU inference.

        Args:
            df: DataFrame with 'processed_posts' column
            sample_n: If set, only process first N rows
            verbose: Print progress details
            skip_reasoning: Skip reasoning LLM call for speed (default True)

        Returns:
            preds: np.ndarray of shape [N, 4] with binary predictions
            results: List of full result dicts
        """
        if sample_n:
            df = df.iloc[:sample_n].copy()

        n = len(df)
        preds = np.zeros((n, 4), dtype=int)
        results = [None] * n

        if self.n_workers >= 2:
            # ── Parallel dual-GPU inference ──
            self._predict_batch_parallel(df, n, preds, results, skip_reasoning, verbose)
        else:
            # ── Sequential single-GPU inference ──
            self._predict_batch_sequential(df, n, preds, results, skip_reasoning, verbose)

        return preds, results

    def _predict_batch_sequential(self, df, n, preds, results, skip_reasoning, verbose):
        """Single-GPU sequential processing."""
        agent = self.agents[0]
        for i in tqdm(range(n), desc='RAG+Agent Prediction'):
            posts = df.iloc[i]['processed_posts']
            try:
                result = agent.run(posts, skip_reasoning=skip_reasoning)
                for j, axis in enumerate(DIM_NAMES):
                    preds[i, j] = result['axes'][axis]
                results[i] = result
            except Exception as e:
                print(f'  ⚠️ Error on sample {i}: {e}')
                results[i] = {
                    'axes': {axis: 0 for axis in DIM_NAMES},
                    'mbti_type': 'ISTP',
                    'reasoning': f'Error: {e}',
                }

            if (i + 1) % 50 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if verbose:
                    mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
                    print(f'  [{i+1}/{n}] VRAM: {mem:.2f}GB')

    def _predict_batch_parallel(self, df, n, preds, results, skip_reasoning, verbose):
        """Dual-GPU parallel processing using threads."""
        completed = [0]  # mutable counter for progress
        progress_lock = threading.Lock()
        pbar = tqdm(total=n, desc=f'RAG+Agent ({self.n_workers} GPUs)')

        def _process_one(idx, agent_idx):
            """Process a single sample on specified GPU agent."""
            agent = self.agents[agent_idx]
            posts = df.iloc[idx]['processed_posts']
            try:
                result = agent.run(posts, skip_reasoning=skip_reasoning)
                for j, axis in enumerate(DIM_NAMES):
                    preds[idx, j] = result['axes'][axis]
                results[idx] = result
            except Exception as e:
                results[idx] = {
                    'axes': {axis: 0 for axis in DIM_NAMES},
                    'mbti_type': 'ISTP',
                    'reasoning': f'Error: {e}',
                }
            with progress_lock:
                completed[0] += 1
                pbar.update(1)

        # Submit tasks — alternate samples across GPUs
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            futures = []
            for i in range(n):
                gpu_idx = i % self.n_workers
                futures.append(executor.submit(_process_one, i, gpu_idx))

            # Wait for all to complete
            for future in as_completed(futures):
                future.result()  # raise any exceptions

        pbar.close()

        # Memory cleanup
        gc.collect()
        if torch.cuda.is_available():
            for gpu_id in range(self.n_workers):
                torch.cuda.empty_cache()

        if verbose:
            for gpu_id in range(min(self.n_workers, NUM_GPUS)):
                mem = torch.cuda.memory_allocated(gpu_id) / 1e9
                print(f'  GPU {gpu_id} VRAM: {mem:.2f}GB')

    def predict_single_with_reasoning(self, user_posts: List[str]) -> dict:
        """Predict MBTI for a single user WITH full reasoning (for demo)."""
        return self.agents[0].run(user_posts, skip_reasoning=False)


# ── Instantiate agents (one per GPU) ──
all_agents = []
for gpu_idx in range(len(llm_models)):
    gpu_llm_fn = make_llm_generate_fn(gpu_idx)
    gpu_predictor = MBTIPredictorTool(gpu_llm_fn, MBTI_KNOWLEDGE_BASE)
    gpu_tools = {
        'retrieve_similar_profiles': tool_retriever,   # shared (CPU/read-only)
        'analyze_linguistic_features': tool_analyzer,   # shared (stateless)
        'lookup_mbti_knowledge': tool_knowledge,        # shared (read-only)
        'predict_mbti': gpu_predictor,                  # GPU-specific
    }
    gpu_agent = MBTIAnalysisAgent(
        tools=gpu_tools,
        llm_fn=gpu_llm_fn,
        confidence_threshold=CONFIG['CONFIDENCE_THRESHOLD'],
        max_reanalyze=CONFIG['MAX_REANALYZE_LOOPS'],
    )
    all_agents.append(gpu_agent)

orchestrator = MultiAgentOrchestrator(all_agents)
print(f'✅ Orchestrator ready — {len(all_agents)} agent(s) on {len(all_agents)} GPU(s)')

# %% [markdown]
# ## 🔬 Cell 11 — Demo: Single User Prediction (Show Agent Trace)

# %%
print('=' * 60)
print('🔬 DEMO: Single User Prediction with Full Agent Trace')
print('=' * 60)

demo_row = test_df.iloc[0]
demo_posts = demo_row['processed_posts']
demo_true_type = demo_row['type']

t0 = time.time()
demo_result = orchestrator.predict_single_with_reasoning(demo_posts)
t1 = time.time()

print(f'\n📝 True MBTI: {demo_true_type}')
print(f'🤖 Predicted:  {demo_result["mbti_type"]}')
print(f'⏱  Time: {t1-t0:.1f}s')
print(f'\n📊 Axis Predictions:')
for axis in DIM_NAMES:
    pred = demo_result['axes'][axis]
    conf = demo_result['confidence'][axis]
    label_map = {'I/E': {0:'I',1:'E'}, 'S/N': {0:'S',1:'N'},
                 'T/F': {0:'T',1:'F'}, 'J/P': {0:'P',1:'J'}}
    print(f'  {axis}: {label_map[axis][pred]} (confidence: {conf:.2f})')

print(f'\n💭 Reasoning:\n{demo_result["reasoning"]}')

print(f'\n🔄 Agent Trace ({len(demo_result["trace"])} steps):')
for step in demo_result['trace']:
    print(f'  {step}')

# %% [markdown]
# ## 🚀 Cell 12 — Run on Full Test Set

# %%
test_sample = CONFIG['TEST_SAMPLE']
test_subset = test_df.iloc[:test_sample].copy() if test_sample else test_df.copy()

n_gpus = len(llm_models)
# With skip_reasoning + dual-GPU: ~8s/sample effective → 4x faster
est_per_sample = 15 / n_gpus  # rough estimate
print(f'⏳ Running RAG+Agent on {len(test_subset)} test samples...')
print(f'   GPUs: {n_gpus} | skip_reasoning=True | est: ~{len(test_subset) * est_per_sample / 60:.0f} min')

t0 = time.time()
rag_preds, rag_results = orchestrator.predict_batch(
    test_subset, sample_n=test_sample, verbose=True, skip_reasoning=True
)
elapsed = time.time() - t0
print(f'\n✅ Inference complete in {elapsed/60:.1f} minutes ({elapsed/len(test_subset):.1f}s/sample)')

# %% [markdown]
# ## 📊 Cell 13 — Evaluation & Save Results

# %%
y_true = test_subset[LABEL_COLS].values[:len(rag_preds)]

# ── Extract per-axis confidence scores for AUC ──
rag_conf = np.zeros((len(rag_results), 4), dtype=float)
for i, res in enumerate(rag_results):
    conf_dict = res.get('confidence', {})
    for j, axis in enumerate(DIM_NAMES):
        rag_conf[i, j] = conf_dict.get(axis, 0.5)

# Convert confidence to probability of class 1:
# If prediction=1, prob=confidence; if prediction=0, prob=1-confidence
rag_prob = np.zeros_like(rag_conf)
for j in range(4):
    rag_prob[:, j] = np.where(
        rag_preds[:, j] == 1,
        rag_conf[:, j],
        1.0 - rag_conf[:, j]
    )

# ── Per-axis metrics ──
print('\n' + '=' * 70)
print('📊 PART 1: Per-Axis Binary Metrics (4 dimensions)')
print('=' * 70)
print(f'{"Axis":<6} {"F1-macro":>10} {"F1-wt":>10} {"Acc":>10} {"AUC":>10}')
print('-' * 50)

axis_scores = {}
for i, (col, name) in enumerate(zip(LABEL_COLS, DIM_NAMES)):
    yt_i, yp_i, yprob_i = y_true[:, i], rag_preds[:, i], rag_prob[:, i]
    f1_mac = f1_score(yt_i, yp_i, average='macro', zero_division=0)
    f1_wt  = f1_score(yt_i, yp_i, average='weighted', zero_division=0)
    acc    = accuracy_score(yt_i, yp_i)
    try:
        auc = roc_auc_score(yt_i, yprob_i)
    except ValueError:
        auc = float('nan')
    axis_scores[name] = {'f1_macro': f1_mac, 'f1_weighted': f1_wt, 'accuracy': acc, 'auc': auc}
    print(f'  {name:<4} {f1_mac:>10.4f} {f1_wt:>10.4f} {acc:>10.4f} {auc:>10.4f}')

avg_f1  = np.mean([s['f1_macro'] for s in axis_scores.values()])
avg_f1w = np.mean([s['f1_weighted'] for s in axis_scores.values()])
avg_acc = np.mean([s['accuracy'] for s in axis_scores.values()])
avg_auc = np.nanmean([s['auc'] for s in axis_scores.values()])
print('-' * 50)
print(f'  {"Avg":<4} {avg_f1:>10.4f} {avg_f1w:>10.4f} {avg_acc:>10.4f} {avg_auc:>10.4f}')

# ── Per-axis detailed classification report ──
print('\n--- Per-Axis Classification Reports ---')
axis_label_names = {
    'I/E': ['I (0)', 'E (1)'],
    'S/N': ['S (0)', 'N (1)'],
    'T/F': ['T (0)', 'F (1)'],
    'J/P': ['P (0)', 'J (1)'],
}
for i, (col, name) in enumerate(zip(LABEL_COLS, DIM_NAMES)):
    print(f'\n  {name}:')
    report = classification_report(
        y_true[:, i], rag_preds[:, i],
        target_names=axis_label_names[name], zero_division=0, digits=4
    )
    for line in report.split('\n'):
        print(f'    {line}')

# ── 16-class overall metrics ──
def preds_to_mbti(preds_row):
    m = {0: {0: 'I', 1: 'E'}, 1: {0: 'S', 1: 'N'},
         2: {0: 'T', 1: 'F'}, 3: {0: 'P', 1: 'J'}}
    return ''.join(m[i][preds_row[i]] for i in range(4))

y_true_types = [preds_to_mbti(y_true[i]) for i in range(len(y_true))]
y_pred_types = [preds_to_mbti(rag_preds[i]) for i in range(len(rag_preds))]

MBTI_LABELS_UPPER = [t.upper() for t in MBTI_TYPES]  # ordered list of 16 types

overall_acc  = accuracy_score(y_true_types, y_pred_types)
overall_f1   = f1_score(y_true_types, y_pred_types, average='macro', zero_division=0)
overall_f1_w = f1_score(y_true_types, y_pred_types, average='weighted', zero_division=0)

# 16-class AUC (one-vs-rest): build probability matrix from 4 axis probs
# P(type) ≈ product of axis probs for that type's configuration
def build_type_prob_matrix(preds_4, prob_4, type_labels):
    """Build [N, 16] probability matrix from 4-axis predictions."""
    axis_configs = {}
    for t in type_labels:
        tl = t.upper()
        axis_configs[t] = [
            int(tl[0] == 'E'),  # I/E
            int(tl[1] == 'N'),  # S/N
            int(tl[2] == 'F'),  # T/F
            int(tl[3] == 'J'),  # J/P
        ]
    n = len(prob_4)
    prob_matrix = np.zeros((n, len(type_labels)), dtype=float)
    for j, t in enumerate(type_labels):
        cfg = axis_configs[t]
        p = np.ones(n)
        for ax in range(4):
            if cfg[ax] == 1:
                p *= prob_4[:, ax]
            else:
                p *= (1.0 - prob_4[:, ax])
        prob_matrix[:, j] = p
    # Normalize rows to sum to 1
    row_sums = prob_matrix.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    prob_matrix = prob_matrix / row_sums
    return prob_matrix

type_prob_matrix = build_type_prob_matrix(rag_preds, rag_prob, MBTI_LABELS_UPPER)

try:
    y_true_bin = label_binarize(y_true_types, classes=MBTI_LABELS_UPPER)
    overall_auc = roc_auc_score(
        y_true_bin, type_prob_matrix, average='macro', multi_class='ovr'
    )
except ValueError:
    overall_auc = float('nan')

print('\n' + '=' * 70)
print('📊 PART 2: Full 16-Type Classification Metrics')
print('=' * 70)
print(f'  Accuracy:     {overall_acc:.4f}')
print(f'  Macro-F1:     {overall_f1:.4f}')
print(f'  Weighted-F1:  {overall_f1_w:.4f}')
print(f'  Macro-AUC:    {overall_auc:.4f}')

# ── Per-type classification report ──
print('\n--- 16-Type Classification Report ---')
type_report = classification_report(
    y_true_types, y_pred_types,
    labels=MBTI_LABELS_UPPER, target_names=MBTI_LABELS_UPPER,
    zero_division=0, digits=4
)
print(type_report)

# ── Per-type support summary ──
from collections import Counter as _Counter
true_dist = _Counter(y_true_types)
pred_dist = _Counter(y_pred_types)
print('--- Type Distribution (True vs Predicted) ---')
print(f'{"Type":<6} {"True":>6} {"Pred":>6} {"Diff":>6}')
for t in MBTI_LABELS_UPPER:
    tc, pc = true_dist.get(t, 0), pred_dist.get(t, 0)
    print(f'  {t:<4} {tc:>6} {pc:>6} {pc - tc:>+6}')

# ── Save predictions (include prob for AUC) ──
save_df = pd.DataFrame()
for i, c in enumerate(LABEL_COLS):
    save_df[f'y_true_{c}'] = y_true[:, i]
    save_df[f'y_pred_{c}'] = rag_preds[:, i]
    save_df[f'y_prob_{c}'] = rag_prob[:, i]

save_path = os.path.join(CONFIG['RESULT_DIR'], 'rag_multi_agent_predictions.csv')
save_df.to_csv(save_path, index=False)
print(f'\n✅ Predictions saved → {save_path}')

# ── Save detailed results with reasoning ──
reasoning_records = []
for i, res in enumerate(rag_results):
    reasoning_records.append({
        'index': i,
        'true_type': test_subset.iloc[i]['type'],
        'pred_type': res.get('mbti_type', ''),
        'reasoning': res.get('reasoning', ''),
        **{f'conf_{axis}': res.get('confidence', {}).get(axis, 0)
           for axis in DIM_NAMES},
    })

reasoning_df = pd.DataFrame(reasoning_records)
reasoning_path = os.path.join(CONFIG['RESULT_DIR'], 'rag_agent_detailed_results.csv')
reasoning_df.to_csv(reasoning_path, index=False)
print(f'✅ Detailed results (with reasoning) saved → {reasoning_path}')

# %% [markdown]
# ## 📈 Cell 14 — Comparison with Baselines

# %%
def load_baseline_results(result_dir):
    """Load all baseline prediction files and compute metrics (F1, Acc, AUC) for both 4-axis and 16-type."""
    files = {
        'SVM + TF-IDF': 'svm_predictions.csv',
        'RoBERTa-mean': 'roberta_predictions.csv',
        'D-DGCN': 'ddgcn_predictions.csv',
    }
    rows = []
    for name, fname in files.items():
        fpath = os.path.join(result_dir, fname)
        if not os.path.exists(fpath):
            continue
        df_r = pd.read_csv(fpath)
        row = {'Model': name}
        f1s, accs, aucs, f1ws = [], [], [], []
        all_yt = np.zeros((len(df_r), 4), dtype=int)
        all_yp = np.zeros((len(df_r), 4), dtype=int)
        all_yprob = np.zeros((len(df_r), 4), dtype=float)
        has_all_axes = True
        for col_i, (col, dim) in enumerate(zip(LABEL_COLS, DIM_NAMES)):
            true_col = f'y_true_{col}'
            pred_col = f'y_pred_{col}'
            if true_col in df_r.columns and pred_col in df_r.columns:
                yt = df_r[true_col].values
                yp = df_r[pred_col].values
                all_yt[:, col_i] = yt
                all_yp[:, col_i] = yp
                f1 = f1_score(yt, yp, average='macro', zero_division=0)
                f1w = f1_score(yt, yp, average='weighted', zero_division=0)
                acc = accuracy_score(yt, yp)
                row[f'F1 {dim}'] = round(f1, 4)
                row[f'Acc {dim}'] = round(acc, 4)
                f1s.append(f1); f1ws.append(f1w); accs.append(acc)
                # AUC
                prob_col = f'y_prob_{col}'
                if prob_col in df_r.columns:
                    yprob = df_r[prob_col].values
                    all_yprob[:, col_i] = yprob
                else:
                    yprob = yp.astype(float)
                    all_yprob[:, col_i] = yprob
                try:
                    auc = roc_auc_score(yt, yprob)
                    row[f'AUC {dim}'] = round(auc, 4)
                    aucs.append(auc)
                except ValueError:
                    row[f'AUC {dim}'] = float('nan')
            else:
                has_all_axes = False
        if f1s:
            row['Avg F1'] = round(np.mean(f1s), 4)
            row['Avg F1-wt'] = round(np.mean(f1ws), 4)
            row['Avg Acc'] = round(np.mean(accs), 4)
            row['Avg AUC'] = round(np.nanmean(aucs), 4) if aucs else float('nan')
        # ── 16-type metrics for baseline ──
        if has_all_axes and len(df_r) > 0:
            bl_true_types = [preds_to_mbti(all_yt[i]) for i in range(len(all_yt))]
            bl_pred_types = [preds_to_mbti(all_yp[i]) for i in range(len(all_yp))]
            row['16-Acc'] = round(accuracy_score(bl_true_types, bl_pred_types), 4)
            row['16-F1'] = round(f1_score(bl_true_types, bl_pred_types, average='macro', zero_division=0), 4)
            row['16-F1-wt'] = round(f1_score(bl_true_types, bl_pred_types, average='weighted', zero_division=0), 4)
            # 16-class AUC
            bl_type_prob = build_type_prob_matrix(all_yp, all_yprob, MBTI_LABELS_UPPER)
            try:
                bl_true_bin = label_binarize(bl_true_types, classes=MBTI_LABELS_UPPER)
                row['16-AUC'] = round(roc_auc_score(bl_true_bin, bl_type_prob, average='macro', multi_class='ovr'), 4)
            except ValueError:
                row['16-AUC'] = float('nan')
        if f1s:
            rows.append(row)
    return rows


# ── Build comparison table ──
baseline_rows = load_baseline_results(CONFIG['RESULT_DIR'])

# Add RAG+Agent results
rag_row = {'Model': '🤖 RAG + Multi-Agent (Ours)'}
for i, dim in enumerate(DIM_NAMES):
    rag_row[f'F1 {dim}'] = round(axis_scores[dim]['f1_macro'], 4)
    rag_row[f'Acc {dim}'] = round(axis_scores[dim]['accuracy'], 4)
    rag_row[f'AUC {dim}'] = round(axis_scores[dim]['auc'], 4)
rag_row['Avg F1'] = round(avg_f1, 4)
rag_row['Avg F1-wt'] = round(avg_f1w, 4)
rag_row['Avg Acc'] = round(avg_acc, 4)
rag_row['Avg AUC'] = round(avg_auc, 4)
rag_row['16-Acc'] = round(overall_acc, 4)
rag_row['16-F1'] = round(overall_f1, 4)
rag_row['16-F1-wt'] = round(overall_f1_w, 4)
rag_row['16-AUC'] = round(overall_auc, 4)

all_rows = baseline_rows + [rag_row]
comparison_df = pd.DataFrame(all_rows).set_index('Model')

# ── Display: 4-axis F1 comparison ──
f1_cols = [f'F1 {d}' for d in DIM_NAMES] + ['Avg F1']
acc_cols = [f'Acc {d}' for d in DIM_NAMES] + ['Avg Acc']
auc_cols = [f'AUC {d}' for d in DIM_NAMES] + ['Avg AUC']

print('\n' + '=' * 70)
print('📊 4-AXIS COMPARISON — Macro-F1')
print('=' * 70)
print(comparison_df[[c for c in f1_cols if c in comparison_df.columns]].to_string())

print('\n' + '=' * 70)
print('📊 4-AXIS COMPARISON — Accuracy')
print('=' * 70)
print(comparison_df[[c for c in acc_cols if c in comparison_df.columns]].to_string())

print('\n' + '=' * 70)
print('📊 4-AXIS COMPARISON — AUC')
print('=' * 70)
print(comparison_df[[c for c in auc_cols if c in comparison_df.columns]].to_string())

# ── Display: 16-type comparison ──
type_cols = ['16-Acc', '16-F1', '16-F1-wt', '16-AUC']
print('\n' + '=' * 70)
print('📊 16-TYPE COMPARISON — Full MBTI Classification')
print('=' * 70)
print(comparison_df[[c for c in type_cols if c in comparison_df.columns]].to_string())
print('=' * 70)

# %% [markdown]
# ## 🎯 Cell 15 — Sample Predictions with Reasoning

# %%
print('\n' + '=' * 70)
print('🎯 SAMPLE PREDICTIONS WITH REASONING')
print('=' * 70)

n_show = min(5, len(rag_results))
for i in range(n_show):
    res = rag_results[i]
    true_type = test_subset.iloc[i]['type']
    match = '✅' if res['mbti_type'] == true_type.upper() else '❌'

    print(f'\n{"─" * 50}')
    print(f'Sample {i+1}: True={true_type.upper()} | Predicted={res["mbti_type"]} {match}')
    print(f'Confidence: {res["confidence"]}')
    print(f'Reasoning: {res["reasoning"][:300]}')

print(f'\n{"─" * 50}')
print(f'Total correct (16-class): {sum(1 for y, p in zip(y_true_types, y_pred_types) if y == p)}/{len(y_true_types)} ({overall_acc:.1%})')

# %% [markdown]
# ## 📝 Summary
#
# ### Architecture Recap: Agent = LLM + Planning + Tool Use + Loop + Memory
#
# | Component | Implementation |
# |-----------|---------------|
# | **LLM** | Qwen2.5-7B-Instruct (4-bit NF4 quantized, ~4.5GB VRAM) |
# | **Planning** | Structured 4-step plan: Linguistics → RAG → Predict (single-shot) → Confidence Recheck |
# | **Tool Use** | 4 tools: RAG Retriever, Linguistic Analyzer, Knowledge Lookup, MBTI Predictor (single-shot) |
# | **Loop** | ReAct cycle with confidence-based re-analysis for uncertain axes (max 2 LLM calls vs 6 previously) |
# | **Memory** | Working Memory (per-user analysis state) + Long-Term Memory (MBTI KB + FAISS) |
#
# ### Key Design Decisions
# - **Dual-embedding RAG**: Raw post embeddings + linguistic feature embeddings for better retrieval
# - **Balanced retrieval**: Ensures minority class representation in few-shot examples
# - **Single-shot prediction**: 1 LLM call predicts all 4 axes (3x faster, captures cross-axis correlations like IN→T)
# - **Confidence re-check loop**: Axes below threshold get re-analyzed with more RAG context (1 extra LLM call)
# - **4-bit quantization**: Enables 7B parameter model on T4 GPU within ~4.5GB VRAM
# - **Keyword masking**: Prevents model from cheating on explicit MBTI mentions
