"""
MBTI Personality Prediction Pipeline  —  v4
============================================
RAG + Multi-Agent Architecture
Dataset: https://www.kaggle.com/datasets/datasnaek/mbti-type/data

Changelog v3 → v4:
  [1] TraitAgent.analyze() — RAG dùng label fraction thay vì mean similarity
      rag_score = #retrieved có left-pole label / top_k  (nhất quán với paper)
  [2] evaluate_full_pipeline() — chạy toàn bộ orchestrator trên test set,
      tính exact-match 16-type từ predict_user() thực sự
  [3] top_k = 5 nhất quán ở mọi nơi (TraitAgent, evaluate_paper_metrics, demo)
  [4] Split 60 / 20 / 20 (train / val / test) — val dùng để tune NS threshold
  [5] Tối ưu riêng trục NS: GridSearch C trên val, threshold tuning trên val
"""

import re
import warnings
warnings.filterwarnings(
    "ignore",
    message=r"`BaseEstimator._validate_data` is deprecated",
    category=FutureWarning,
)
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from scipy.sparse import hstack, csr_matrix

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, f1_score,
    accuracy_score, roc_auc_score,
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False
    warnings.warn(
        "imbalanced-learn not found — install with: pip install imbalanced-learn. "
        "SMOTE will be skipped."
    )

# ═══════════════════════════════════════════════
# 1. CONSTANTS
# ═══════════════════════════════════════════════

MBTI_WORDS = [
    "intj", "intp", "entj", "entp",
    "infj", "infp", "enfj", "enfp",
    "istj", "isfj", "estj", "esfj",
    "istp", "isfp", "estp", "esfp",
    "mbti",
]

AXIS_LABELS: Dict[str, Tuple[str, str]] = {
    "IE": ("I", "E"),
    "NS": ("N", "S"),
    "TF": ("T", "F"),
    "JP": ("J", "P"),
}

# [3] Unified top_k — used everywhere: TraitAgent, paper_metrics, full_pipeline
TOP_K = 5

# Fusion weight: fused = (1-ALPHA)*model_prob + ALPHA*rag_label_frac
ALPHA = 0.20

# Axes that get SMOTE (most imbalanced)
SMOTE_AXES = {"IE", "NS"}

# [5] NS-specific: C values searched on val set
NS_C_CANDIDATES = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]

# ═══════════════════════════════════════════════
# 2. PER-AXIS LEXICONS
# ═══════════════════════════════════════════════

LEXICONS: Dict[str, Dict[str, List[str]]] = {
    "IE": {
        "I": ["alone", "quiet", "introvert", "solitude", "reserved", "private",
              "shy", "reclusive", "reflective", "recharge", "tired", "drained",
              "myself", "inner", "silent", "book", "home"],
        "E": ["people", "social", "party", "friends", "outgoing", "extrovert",
              "talkative", "energized", "crowd", "network", "lively", "fun",
              "group", "together", "meet", "exciting"],
    },
    "NS": {
        "N": ["imagine", "abstract", "theory", "concept", "idea", "pattern",
              "future", "possibility", "creative", "intuition", "meaning",
              "insight", "vision", "philosophy", "metaphor", "symbolic"],
        "S": ["practical", "concrete", "detail", "fact", "reality", "experience",
              "sensory", "present", "specific", "observable", "routine",
              "literal", "hands", "tangible", "physical", "traditional"],
    },
    "TF": {
        "T": ["logic", "reason", "analysis", "objective", "principle", "criteria",
              "efficient", "competence", "rational", "consistent", "fair",
              "impersonal", "systematic", "critique", "truth", "argument"],
        "F": ["feel", "emotion", "empathy", "value", "harmony", "compassion",
              "hurt", "care", "personal", "relationship", "kind", "warm",
              "understand", "support", "heart", "love"],
    },
    "JP": {
        "J": ["plan", "schedule", "organised", "structure", "deadline", "list",
              "decided", "closure", "systematic", "control", "prepare", "goal",
              "routine", "punctual", "tidy", "certain"],
        "P": ["flexible", "spontaneous", "adapt", "open", "option", "freedom",
              "explore", "improvise", "curious", "casual", "wander", "change",
              "last minute", "undecided", "variety", "discover"],
    },
}


def lexicon_features(texts: List[str]) -> np.ndarray:
    """8-dim array: [I,E,N,S,T,F,J,P] word-count normalised by text length."""
    rows = []
    for text in texts:
        words = text.split()
        n = max(len(words), 1)
        feats = []
        for axis, poles in LEXICONS.items():
            left_label, right_label = AXIS_LABELS[axis]
            feats.append(sum(1 for w in words if w in poles[left_label])  / n)
            feats.append(sum(1 for w in words if w in poles[right_label]) / n)
        rows.append(feats)
    return np.array(rows, dtype=np.float32)

# ═══════════════════════════════════════════════
# 3. TEXT PREPROCESSING
# ═══════════════════════════════════════════════

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\|\|\|", " ", text)
    text = re.sub(r"\d+", " ", text)
    for w in MBTI_WORDS:
        text = re.sub(rf"\b{w}\b", " ", text)
    text = re.sub(r"[^a-zA-Z\s!?]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_posts(raw_posts: str) -> List[str]:
    return [clean_text(p) for p in raw_posts.split("|||") if p.strip()]


def mbti_to_axes(mbti_type: str) -> Dict[str, int]:
    return {
        "IE": 1 if mbti_type[0] == "I" else 0,
        "NS": 1 if mbti_type[1] == "N" else 0,
        "TF": 1 if mbti_type[2] == "T" else 0,
        "JP": 1 if mbti_type[3] == "J" else 0,
    }

# ═══════════════════════════════════════════════
# 4. DATA LOADING
# ═══════════════════════════════════════════════

def load_dataset(path: str = "dataset/mbti_1.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df["clean_text"] = df["posts"].apply(clean_text)
    df["post_list"]  = df["posts"].apply(split_posts)
    for axis in AXIS_LABELS:
        df[axis] = df["type"].apply(lambda x: mbti_to_axes(x)[axis])
    return df

# ═══════════════════════════════════════════════
# 5. [CHANGE 4] SPLIT 60 / 20 / 20
# ═══════════════════════════════════════════════

def make_splits(
    df: pd.DataFrame,
    val_size:  float = 0.20,
    test_size: float = 0.20,
    random_state: int = 42,
):
    """
    [4] 60/20/20 split: train / val / test.
    All three sets share the same index ordering to prevent X/y misalignment.
    val set is used exclusively for NS threshold tuning (see tune_ns_threshold).
    """
    idx = np.arange(len(df))

    # First carve out test (20%)
    rest_idx, test_idx = train_test_split(
        idx, test_size=test_size, random_state=random_state
    )
    # Then carve out val from the remaining 80% → val = 20/80 = 0.25 of rest
    val_frac = val_size / (1.0 - test_size)
    train_idx, val_idx = train_test_split(
        rest_idx, test_size=val_frac, random_state=random_state
    )

    def _subset(col: str, idx_arr) -> pd.Series:
        return df[col].iloc[idx_arr].reset_index(drop=True)

    X_train = _subset("clean_text", train_idx)
    X_val   = _subset("clean_text", val_idx)
    X_test  = _subset("clean_text", test_idx)

    y_splits: Dict[str, Tuple] = {}
    for axis in AXIS_LABELS:
        y_splits[axis] = (
            _subset(axis, train_idx),
            _subset(axis, val_idx),
            _subset(axis, test_idx),
        )

    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_val   = df.iloc[val_idx].reset_index(drop=True)
    df_test  = df.iloc[test_idx].reset_index(drop=True)

    print(f"  Train: {len(train_idx)}  Val: {len(val_idx)}  Test: {len(test_idx)}")
    return X_train, X_val, X_test, y_splits, df_train, df_val, df_test

# ═══════════════════════════════════════════════
# 6. FEATURE EXTRACTION
# ═══════════════════════════════════════════════

class FeatureBuilder:
    """word TF-IDF (15k bigram) + char TF-IDF (10k 2–4) + lexicon (8 dims)."""

    def __init__(self, word_max: int = 15_000, char_max: int = 10_000):
        self.word_tfidf = TfidfVectorizer(
            max_features=word_max, stop_words="english",
            ngram_range=(1, 2), sublinear_tf=True, analyzer="word",
        )
        self.char_tfidf = TfidfVectorizer(
            max_features=char_max, ngram_range=(2, 4),
            sublinear_tf=True, analyzer="char_wb",
        )

    def fit_transform(self, texts: List[str]):
        W = self.word_tfidf.fit_transform(texts)
        C = self.char_tfidf.fit_transform(texts)
        L = csr_matrix(lexicon_features(texts))
        return hstack([W, C, L])

    def transform(self, texts: List[str]):
        W = self.word_tfidf.transform(texts)
        C = self.char_tfidf.transform(texts)
        L = csr_matrix(lexicon_features(texts))
        return hstack([W, C, L])

# ═══════════════════════════════════════════════
# 7. [CHANGE 5] NS OPTIMISATION HELPERS
# ═══════════════════════════════════════════════

def _tune_ns_C(
    X_train_vec, y_train_ns,
    X_val_vec,   y_val_ns,
) -> float:
    """
    [5] Grid-search C on val set for NS axis using macro-F1 as criterion.
    NS is the hardest axis (86%/14% imbalance) so deserves its own C.
    """
    print("  [NS] Searching best C on val set …")
    best_c, best_f1 = 1.0, 0.0
    for c in NS_C_CANDIDATES:
        clf = LogisticRegression(
            class_weight="balanced", max_iter=1000,
            C=c, solver="lbfgs", random_state=42,
        )
        # Use SMOTE on train for this trial too
        X_fit, y_fit = X_train_vec, y_train_ns
        if HAS_SMOTE:
            try:
                sm = SMOTE(random_state=42, k_neighbors=5)
                X_fit, y_fit = sm.fit_resample(X_train_vec, y_train_ns)
            except Exception:
                pass
        clf.fit(X_fit, y_fit)
        f1 = f1_score(y_val_ns, clf.predict(X_val_vec), average="macro")
        print(f"    C={c:<6}  val macro-F1={f1:.4f}")
        if f1 > best_f1:
            best_f1, best_c = f1, c
    print(f"  [NS] Best C={best_c}  val macro-F1={best_f1:.4f}")
    return best_c


def _tune_ns_threshold(
    clf_ns, X_val_vec, y_val_ns
) -> float:
    """
    [5] Find the decision threshold t ∈ [0.1, 0.9] on val set that maximises
    macro-F1 for NS.  Default is 0.5 but NS minority (S=14%) benefits from
    shifting t toward the majority pole.
    """
    proba = clf_ns.predict_proba(X_val_vec)[:, 1]   # P(N)
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.1, 0.91, 0.05):
        pred = (proba >= t).astype(int)
        f1 = f1_score(y_val_ns, pred, average="macro")
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    print(f"  [NS] Best threshold={best_t:.2f}  val macro-F1={best_f1:.4f}")
    return best_t

# ═══════════════════════════════════════════════
# 8. MODEL TRAINING
# ═══════════════════════════════════════════════

def train_classifiers(
    X_train_vec,
    y_splits:   Dict[str, Tuple],
    X_val_vec   = None,
    C:          float = 1.0,
) -> Tuple[Dict[str, LogisticRegression], Dict[str, float]]:
    """
    Train 4 binary classifiers.
    [5] NS gets its own C (tuned on val) and its own threshold.
    Returns (clfs, thresholds) — thresholds[axis] is the decision threshold.
    """
    clfs:       Dict[str, LogisticRegression] = {}
    thresholds: Dict[str, float] = {ax: 0.5 for ax in AXIS_LABELS}

    for axis in AXIS_LABELS:
        y_train, y_val, _ = y_splits[axis]

        # [5] NS-specific C search
        if axis == "NS" and X_val_vec is not None:
            best_c = _tune_ns_C(X_train_vec, y_train, X_val_vec, y_val)
        else:
            best_c = C

        X_fit, y_fit = X_train_vec, y_train
        if HAS_SMOTE and axis in SMOTE_AXES:
            try:
                sm = SMOTE(random_state=42, k_neighbors=5)
                X_fit, y_fit = sm.fit_resample(X_train_vec, y_train)
                print(f"  [{axis}] SMOTE: {len(y_train)} → {len(y_fit)} samples")
            except Exception as e:
                print(f"  [{axis}] SMOTE skipped ({e})")

        clf = LogisticRegression(
            class_weight="balanced", max_iter=1000,
            C=best_c, solver="lbfgs", random_state=42,
        )
        clf.fit(X_fit, y_fit)
        clfs[axis] = clf

        # [5] NS threshold tuning on val
        if axis == "NS" and X_val_vec is not None:
            thresholds["NS"] = _tune_ns_threshold(clf, X_val_vec, y_val)

    return clfs, thresholds

# ═══════════════════════════════════════════════
# 9. RAG KNOWLEDGE BASE
# ═══════════════════════════════════════════════

class KnowledgeBase:
    """
    Retrieval index built from training data only.
    retrieve(user_text, axis) blends user text + axis seed → personalised evidence.
    """

    AXIS_SEEDS = {
        "IE": "alone quiet social energy group introvert extrovert",
        "NS": "abstract imagine theory pattern future concrete fact detail sensory",
        "TF": "logic reason objective empathy emotion value feeling analysis",
        "JP": "plan schedule organised flexible spontaneous structure deadline",
    }

    # Position of each axis character in the MBTI string
    AXIS_POS = {"IE": 0, "NS": 1, "TF": 2, "JP": 3}

    def __init__(self, df_train: pd.DataFrame, max_features: int = 8000):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features, stop_words="english",
            sublinear_tf=True, ngram_range=(1, 2),
        )
        self.texts:  List[str] = df_train["clean_text"].tolist()
        self.labels: List[str] = df_train["type"].tolist()
        self.matrix = self.vectorizer.fit_transform(self.texts)

    def retrieve(
        self,
        user_text: str,
        axis: str,
        top_k: int = TOP_K,
        user_weight: float = 0.7,
    ) -> List[Dict]:
        """
        Blend user text vector + axis seed vector, then retrieve top_k.
        Returns list of dicts with keys: text, label, score, rank.
        """
        seed   = self.AXIS_SEEDS.get(axis, "")
        u_vec  = normalize(self.vectorizer.transform([user_text]))
        s_vec  = normalize(self.vectorizer.transform([seed]))
        q      = user_weight * u_vec + (1.0 - user_weight) * s_vec

        sims = cosine_similarity(q, self.matrix)[0]
        idx  = np.argsort(sims)[::-1][:top_k]

        return [
            {"text": self.texts[i][:200], "label": self.labels[i],
             "score": float(sims[i]), "rank": int(r)}
            for r, i in enumerate(idx)
        ]

    def label_fraction(
        self, user_text: str, axis: str, top_k: int = TOP_K
    ) -> float:
        """
        [1] Fraction of top_k retrieved neighbours whose MBTI has the left-pole
        character at the correct axis position.
        e.g. axis=IE → count labels[0]=='I' → returns P(left=I | evidence).
        This is exactly the RAG-probability approximation described in the paper.
        """
        evidence   = self.retrieve(user_text, axis=axis, top_k=top_k)
        if not evidence:
            return 0.5
        pole_char  = AXIS_LABELS[axis][0]             # left pole: I, N, T, J
        pos        = self.AXIS_POS[axis]
        left_count = sum(
            1 for e in evidence
            if len(e["label"]) > pos and e["label"][pos] == pole_char
        )
        return left_count / len(evidence)

# ═══════════════════════════════════════════════
# 10. AGENT DATA CLASS
# ═══════════════════════════════════════════════

@dataclass
class AgentResult:
    axis:        str
    left_label:  str
    right_label: str
    model_prob:  float    # P(left) from classifier
    rag_frac:    float    # [1] label fraction from RAG (not mean similarity)
    fused_prob:  float    # (1-ALPHA)*model_prob + ALPHA*rag_frac
    decision:    str
    threshold:   float    # decision threshold (0.5 default, tuned for NS)
    evidence:    List[Dict] = field(default_factory=list)
    rationale:   str = ""

# ═══════════════════════════════════════════════
# 11. TRAIT AGENTS
# ═══════════════════════════════════════════════

class TraitAgent:
    def __init__(
        self,
        axis:            str,
        left_label:      str,
        right_label:     str,
        classifier:      LogisticRegression,
        feature_builder: FeatureBuilder,
        kb:              KnowledgeBase,
        threshold:       float = 0.5,
    ):
        self.axis            = axis
        self.left_label      = left_label
        self.right_label     = right_label
        self.clf             = classifier
        self.feature_builder = feature_builder
        self.kb              = kb
        self.threshold       = threshold   # [5] tuned per axis

    def analyze(self, user_text: str, top_k: int = TOP_K) -> AgentResult:
        # Step 1: classifier probability
        x_vec      = self.feature_builder.transform([user_text])
        model_prob = float(self.clf.predict_proba(x_vec)[0][1])

        # Step 2: [1] RAG label fraction (paper-consistent)
        evidence  = self.kb.retrieve(user_text, axis=self.axis, top_k=top_k)
        pole_char = AXIS_LABELS[self.axis][0]
        pos       = KnowledgeBase.AXIS_POS[self.axis]
        rag_frac  = (
            sum(1 for e in evidence
                if len(e["label"]) > pos and e["label"][pos] == pole_char)
            / len(evidence)
            if evidence else 0.5
        )

        # Step 3: fuse
        fused_prob = float(np.clip(
            (1.0 - ALPHA) * model_prob + ALPHA * rag_frac,
            1e-6, 1.0 - 1e-6,
        ))

        # [5] use per-axis threshold
        decision = self.left_label if fused_prob >= self.threshold else self.right_label

        rationale = (
            f"[{self.axis}] model={model_prob:.3f}  "
            f"rag_frac={rag_frac:.2f}  "
            f"fused={fused_prob:.3f}  "
            f"thr={self.threshold:.2f}  → {decision}"
        )

        return AgentResult(
            axis=self.axis,
            left_label=self.left_label,
            right_label=self.right_label,
            model_prob=model_prob,
            rag_frac=rag_frac,
            fused_prob=fused_prob,
            decision=decision,
            threshold=self.threshold,
            evidence=evidence,
            rationale=rationale,
        )

# ═══════════════════════════════════════════════
# 12. JUDGE AGENT
# ═══════════════════════════════════════════════

def binary_entropy(p: float) -> float:
    p = float(np.clip(p, 1e-9, 1 - 1e-9))
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


class JudgeAgent:
    def combine(self, results: List[AgentResult]) -> Dict:
        mapping       = {r.axis: r for r in results}
        mbti          = "".join(mapping[ax].decision for ax in ["IE", "NS", "TF", "JP"])
        entropies     = [binary_entropy(mapping[ax].fused_prob) for ax in AXIS_LABELS]
        confidence    = round(1.0 - float(np.mean(entropies)), 4)
        per_axis_conf = {
            ax: round(1.0 - binary_entropy(mapping[ax].fused_prob), 4)
            for ax in AXIS_LABELS
        }
        return {
            "mbti_prediction":     mbti,
            "confidence":          confidence,
            "per_axis_confidence": per_axis_conf,
            "details":             results,
        }

# ═══════════════════════════════════════════════
# 13. ORCHESTRATOR
# ═══════════════════════════════════════════════

class MBTIOrchestrator:
    def __init__(self):
        self.feature_builder: Optional[FeatureBuilder]      = None
        self.clfs:            Dict[str, LogisticRegression] = {}
        self.kb:              Optional[KnowledgeBase]       = None
        self.agents:          Dict[str, TraitAgent]         = {}
        self.thresholds:      Dict[str, float]              = {}
        self.judge = JudgeAgent()

    def build(
        self,
        feature_builder: FeatureBuilder,
        clfs:       Dict[str, LogisticRegression],
        kb:         KnowledgeBase,
        thresholds: Dict[str, float],
    ) -> "MBTIOrchestrator":
        self.feature_builder = feature_builder
        self.clfs       = clfs
        self.kb         = kb
        self.thresholds = thresholds
        for axis, (left, right) in AXIS_LABELS.items():
            self.agents[axis] = TraitAgent(
                axis=axis, left_label=left, right_label=right,
                classifier=clfs[axis],
                feature_builder=feature_builder,
                kb=kb,
                threshold=thresholds.get(axis, 0.5),
            )
        return self

    def predict_user(self, raw_posts: str, top_k: int = TOP_K) -> Dict:
        if not self.agents:
            raise RuntimeError("Call build() before predict_user().")
        user_text = clean_text(raw_posts)
        results   = [agent.analyze(user_text, top_k=top_k) for agent in self.agents.values()]
        return self.judge.combine(results)

# ═══════════════════════════════════════════════
# 14. EVALUATION
# ═══════════════════════════════════════════════

def evaluate_classifiers(clfs, thresholds, X_test_vec, y_splits):
    """Detailed per-class classification report for each axis."""
    print("=" * 62)
    print("DETAILED CLASSIFICATION REPORT")
    print("=" * 62)
    for axis, (left, right) in AXIS_LABELS.items():
        _, _, y_test = y_splits[axis]
        t      = thresholds.get(axis, 0.5)
        proba  = clfs[axis].predict_proba(X_test_vec)[:, 1]
        y_pred = (proba >= t).astype(int)
        mf1    = f1_score(y_test, y_pred, average="macro")
        print(f"\n=== {axis}  ({left}/{right})  thr={t:.2f}  macro-F1={mf1:.3f} ===")
        print(classification_report(y_test, y_pred, target_names=[left, right]))


def evaluate_paper_metrics(
    clfs,
    thresholds: Dict[str, float],
    X_test_vec,
    y_splits:   Dict[str, Tuple],
    df_test:    pd.DataFrame,
    kb:         KnowledgeBase,
    top_k:      int = TOP_K,       # [3] unified top_k
) -> Dict:
    """
    Paper-style table: Accuracy | macro-F1 | Clf-AUC | RAG-AUC
    RAG-AUC uses label_fraction as probability proxy (paper method).
    Clf-AUC uses clf.predict_proba for comparison.
    """
    print("\n" + "=" * 72)
    print("PAPER-STYLE EVALUATION TABLE  (ref: arXiv:2509.04461)")
    print(f"Metrics: Accuracy | macro-F1 | Clf-AUC | RAG-AUC   [top_k={top_k}]")
    print("=" * 72)

    print(f"  Computing RAG label fractions for {len(df_test)} test users × 4 axes …")
    rag_probs: Dict[str, np.ndarray] = {ax: np.zeros(len(df_test)) for ax in AXIS_LABELS}
    for i, row in enumerate(df_test.itertuples(index=False)):
        for axis in AXIS_LABELS:
            rag_probs[axis][i] = kb.label_fraction(row.clean_text, axis=axis, top_k=top_k)

    header = f"{'Axis':<6} {'Acc':>7} {'macro-F1':>9} {'Clf-AUC':>9} {'RAG-AUC':>9}  {'thr':>5}"
    print("\n" + header)
    print("-" * len(header))

    acc_list, f1_list, clf_auc_list, rag_auc_list = [], [], [], []
    per_axis_preds: Dict[str, np.ndarray] = {}

    for axis in ["IE", "NS", "TF", "JP"]:
        left, right = AXIS_LABELS[axis]
        _, _, y_test = y_splits[axis]
        t      = thresholds.get(axis, 0.5)
        proba  = clfs[axis].predict_proba(X_test_vec)[:, 1]
        y_pred = (proba >= t).astype(int)
        per_axis_preds[axis] = y_pred

        acc    = accuracy_score(y_test, y_pred)
        mf1    = f1_score(y_test, y_pred, average="macro")
        try:    clf_auc = roc_auc_score(y_test, proba)
        except: clf_auc = float("nan")
        try:    rag_auc = roc_auc_score(y_test, rag_probs[axis])
        except: rag_auc = float("nan")

        acc_list.append(acc)
        f1_list.append(mf1)
        clf_auc_list.append(clf_auc)
        rag_auc_list.append(rag_auc)

        print(
            f"{axis:<6} {acc:>7.4f} {mf1:>9.4f} {clf_auc:>9.4f} {rag_auc:>9.4f}"
            f"  {t:>5.2f}   ({left}/{right})"
        )

    avg = {k: float(np.mean(v)) for k, v in
           [("acc", acc_list), ("f1", f1_list),
            ("clf_auc", clf_auc_list), ("rag_auc", rag_auc_list)]}
    print("-" * len(header))
    print(f"{'Avg':<6} {avg['acc']:>7.4f} {avg['f1']:>9.4f} "
          f"{avg['clf_auc']:>9.4f} {avg['rag_auc']:>9.4f}")

    # 16-type exact match from per-axis classifier predictions
    left_lbl  = {ax: AXIS_LABELS[ax][0] for ax in AXIS_LABELS}
    right_lbl = {ax: AXIS_LABELS[ax][1] for ax in AXIS_LABELS}
    pred_types = [
        "".join(left_lbl[ax] if per_axis_preds[ax][i] == 1 else right_lbl[ax]
                for ax in ["IE", "NS", "TF", "JP"])
        for i in range(len(df_test))
    ]
    exact_clf = sum(p == t for p, t in zip(pred_types, df_test["type"])) / len(df_test)
    print(f"\n{'=' * 72}")
    print(f"16-type exact match (clf-only) : {exact_clf:.4f}  ({exact_clf*100:.1f}%)")
    print(f"{'=' * 72}")

    return {
        "per_axis": {
            ax: {"acc": acc_list[i], "f1": f1_list[i],
                 "clf_auc": clf_auc_list[i], "rag_auc": rag_auc_list[i]}
            for i, ax in enumerate(["IE", "NS", "TF", "JP"])
        },
        "avg": avg,
        "exact_match_clf": exact_clf,
    }


def evaluate_full_pipeline(
    system:   MBTIOrchestrator,
    df_test:  pd.DataFrame,
    top_k:    int = TOP_K,       # [3] unified top_k
) -> Dict:
    """
    [2] Run predict_user() on every test user via the full orchestrator
    (classifier + RAG fusion + judge) and compute:
      • 16-type exact match
      • Per-axis accuracy (using agent decisions, which include RAG fusion + threshold)
      • Mean confidence
    This is the true end-to-end metric, distinct from clf-only evaluation.
    """
    print(f"\n  Running full pipeline on {len(df_test)} test users (top_k={top_k}) …")
    pred_types  = []
    confidences = []

    for i, row in enumerate(df_test.itertuples(index=False)):
        out = system.predict_user(row.posts, top_k=top_k)
        pred_types.append(out["mbti_prediction"])
        confidences.append(out["confidence"])
        if (i + 1) % 200 == 0:
            print(f"    {i+1}/{len(df_test)} …")

    true_types  = df_test["type"].tolist()
    exact_match = sum(p == t for p, t in zip(pred_types, true_types)) / len(true_types)
    mean_conf   = float(np.mean(confidences))

    # Per-axis accuracy from orchestrator decisions
    axis_pos  = {"IE": 0, "NS": 1, "TF": 2, "JP": 3}
    per_axis_acc = {}
    for ax, pos in axis_pos.items():
        left_char  = AXIS_LABELS[ax][0]
        pred_left  = [1 if p[pos] == left_char else 0 for p in pred_types]
        true_left  = [1 if t[pos] == left_char else 0 for t in true_types]
        per_axis_acc[ax] = accuracy_score(true_left, pred_left)

    print(f"\n{'=' * 62}")
    print(f"FULL PIPELINE EVALUATION  (clf + RAG + judge + threshold)")
    print(f"{'=' * 62}")
    print(f"  16-type exact match : {exact_match:.4f}  ({exact_match*100:.1f}%)")
    print(f"  Mean confidence     : {mean_conf:.4f}")
    print(f"  Per-axis accuracy   :", {ax: f"{v:.4f}" for ax, v in per_axis_acc.items()})
    print(f"{'=' * 62}")

    return {
        "exact_match_pipeline": exact_match,
        "mean_confidence": mean_conf,
        "per_axis_accuracy": per_axis_acc,
        "pred_types": pred_types,
        "true_types": true_types,
    }

# ═══════════════════════════════════════════════
# 15. MAIN
# ═══════════════════════════════════════════════

def main():
    print("Loading dataset …")
    df = load_dataset("dataset/mbti_1.csv")
    print(f"  Total users : {len(df)}")

    # [4] 60/20/20 split
    print("\nSplitting data (60 / 20 / 20) …")
    X_train, X_val, X_test, y_splits, df_train, df_val, df_test = make_splits(df)

    print("\nBuilding feature matrix (word-TF-IDF + char-TF-IDF + lexicon) …")
    fb = FeatureBuilder()
    X_train_vec = fb.fit_transform(X_train.tolist())
    X_val_vec   = fb.transform(X_val.tolist())
    X_test_vec  = fb.transform(X_test.tolist())
    print(f"  Feature shape: {X_train_vec.shape}")

    print("\nTraining classifiers …")
    clfs, thresholds = train_classifiers(
        X_train_vec, y_splits, X_val_vec=X_val_vec
    )
    print(f"  Thresholds: {thresholds}")

    print("\nBuilding knowledge base from training data …")
    kb = KnowledgeBase(df_train)

    # ── Evaluation ────────────────────────────────────────────────────────
    evaluate_classifiers(clfs, thresholds, X_test_vec, y_splits)

    evaluate_paper_metrics(
        clfs=clfs, thresholds=thresholds,
        X_test_vec=X_test_vec, y_splits=y_splits,
        df_test=df_test, kb=kb, top_k=TOP_K,
    )

    # ── Full pipeline ─────────────────────────────────────────────────────
    print("\nAssembling orchestrator …")
    system = MBTIOrchestrator().build(
        feature_builder=fb, clfs=clfs, kb=kb, thresholds=thresholds
    )

    # [2] Exact match from orchestrator (true end-to-end metric)
    pipeline_results = evaluate_full_pipeline(system, df_test, top_k=TOP_K)

    # ── Demo ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print(f"DEMO — 3 test users  (top_k={TOP_K})")
    print("=" * 62)

    for idx in range(3):
        row   = df_test.iloc[idx]
        out   = system.predict_user(row["posts"], top_k=TOP_K)
        pred  = out["mbti_prediction"]
        true  = row["type"]
        match = "✓" if pred == true else "✗"

        print(f"\n[User #{idx}]  True: {true}  Predicted: {pred}  {match}")
        print(f"  Confidence : {out['confidence']:.4f}")
        print(f"  Per-axis   : {out['per_axis_confidence']}")
        for d in out["details"]:
            print(f"  {d.rationale}")
            for e in d.evidence[:2]:
                print(f"    [{e['label']}] score={e['score']:.3f}  \"{e['text'][:100]}\"")


if __name__ == "__main__":
    main()