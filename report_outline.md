# OUTLINE BÁO CÁO: MBTI Personality Detection — So sánh Baseline và RAG + Agent

---

## Chương 1: Giới thiệu (Introduction)

### 1.1 Đặt vấn đề
- Bài toán: Nhận diện tính cách MBTI từ văn bản mạng xã hội (social media posts)
- Ứng dụng thực tiễn: tuyển dụng, tâm lý học, hệ thống gợi ý cá nhân hóa
- Thách thức: 16 class (mất cân bằng nghiêm trọng), ngôn ngữ tự nhiên không cấu trúc, overlap giữa các type

### 1.2 Mục tiêu nghiên cứu
- So sánh 3 hướng tiếp cận: Học máy cổ điển (SVM), Học sâu (RoBERTa, D-DGCN), LLM-based Agent (RAG + Agent)
- Đánh giá trên 4 trục nhị phân (I/E, S/N, T/F, J/P) + 16-type
- Metrics: F1-Macro, Accuracy, AUC-ROC

### 1.3 Phạm vi và đóng góp
- Phân rã bài toán 16 class → 4 bài toán nhị phân độc lập
- Đề xuất kiến trúc RAG + Single Agent với Function Calling cho bài toán phân loại tính cách
- So sánh có hệ thống giữa 4 phương pháp trên cùng tập dữ liệu và điều kiện thí nghiệm

---

## Chương 2: Tổng quan lý thuyết (Related Work)

### 2.1 Mô hình MBTI và đặc trưng ngôn ngữ
- 4 chiều nhị phân: I/E, S/N, T/F, J/P → 16 type
- Các đặc trưng ngôn ngữ tâm lý (psycholinguistic features) liên quan đến từng chiều

### 2.2 Các phương pháp phân loại MBTI trong nghiên cứu trước
- Học máy cổ điển: TF-IDF + SVM, Naive Bayes, Random Forest
- Học sâu: BERT, RoBERTa, các biến thể fine-tuning
- Graph Neural Networks: mô hình hóa quan hệ giữa các bài viết (D-DGCN)
- LLM-based: zero-shot / few-shot classification

### 2.3 Retrieval-Augmented Generation (RAG)
- Kiến trúc RAG: Retriever (FAISS, dense embedding) + Generator (LLM)
- Ưu điểm: bổ sung kiến thức ngoài mà không cần fine-tune

### 2.4 LLM Agent
- Khái niệm Agent: LLM + Planning + Tool Use + Loop + Memory
- ReAct framework: Thought → Action → Observation loop
- Function Calling / Tool Binding: LLM tự sinh lệnh gọi tool có cấu trúc

---

## Chương 3: Dữ liệu và Tiền xử lý (Dataset & Preprocessing)

### 3.1 Mô tả dữ liệu
- **MBTI-500 dataset**: ~106K dòng, mỗi dòng = 1 user + MBTI type + posts
- **Kaggle MBTI dataset** (fallback): ~8,675 users, mỗi user có ~50 bài viết (phân tách bằng `|||`)
- Phân bố class: thống kê số lượng mỗi type, visualize histogram (chú ý mất cân bằng)

### 3.2 Tiền xử lý
- **Psycholinguistic masking**: thay thế từ khóa MBTI (infj, introvert, sensing...) → `<type>` để tránh data leakage
- **Truncation**: tối đa 50 posts × 70 words/post
- **Concatenation**: ghép các posts thành chuỗi duy nhất cho mỗi user

### 3.3 Phân chia tập dữ liệu
- **60/20/20** (Train/Val/Test), stratified trên 4 trục đồng thời (`combo_label`)
- `random_state=42` để tái tạo kết quả
- Thống kê kích thước: Train=5,205 | Val=1,735 | Test=1,735

### 3.4 Xử lý mất cân bằng class
- `compute_class_weight('balanced')` → per-axis weights
- Áp dụng `pos_weight` trong BCEWithLogitsLoss (cho mô hình học sâu)
- SVM: `class_weight='balanced'`

---

## Chương 4: Phương pháp đề xuất (Methodology)

### 4.1 Tổng quan kiến trúc thí nghiệm
- Bảng so sánh 4 phương pháp: SVM, RoBERTa, D-DGCN, RAG+Agent
- Pipeline chung: Text → Features/Embeddings → Model → 4 binary predictions → 16-type

### 4.2 Baseline 1: SVM + TF-IDF (Học máy cổ điển)

#### 4.2.1 Kiến trúc
- TF-IDF Vectorizer → Linear SVM (per-axis) → MultiOutputClassifier

#### 4.2.2 Tham số mô hình ⚠️
| Tham số | Giá trị | Giải thích |
|---------|---------|------------|
| `max_features` | 50,000 | Số lượng n-gram tối đa trong vocabulary |
| `ngram_range` | (1, 2) | Unigram + Bigram |
| `sublinear_tf` | True | Áp dụng log scaling cho TF |
| `kernel` | linear | Phân tách tuyến tính trong không gian TF-IDF |
| `C` | 1.0 | Regularization (trade-off margin vs errors) |
| `class_weight` | balanced | Tự động điều chỉnh theo tần suất class |

#### 4.2.3 Đường học (Learning Curve)
- SVM không có quá trình học iterative → không có learning curve theo epoch
- **Thay thế**: Trình bày decision boundary analysis hoặc confusion matrix per axis
- Giải thích: SVM tìm siêu phẳng tối ưu trong 1 bước giải bài toán tối ưu convex → không cần chọn điểm dừng

### 4.3 Baseline 2: RoBERTa-mean (Học sâu - Transformer)

#### 4.3.1 Kiến trúc
- Input: [B, 50 posts, 128 tokens] → RoBERTa encoder → CLS pooling → Mean-pool over 50 posts → 4 linear classification heads
- Mỗi head: `Linear(768, 1)` → sigmoid → binary prediction per axis

#### 4.3.2 Tham số mô hình và thiết lập huấn luyện ⚠️
| Tham số | Giá trị | Giải thích |
|---------|---------|------------|
| Backbone | `roberta-base` | 125M params, 768-dim hidden |
| Max token length | 128 | Cắt mỗi post tại 128 tokens |
| Batch size | 8 | Giới hạn bởi GPU memory (T4 16GB) |
| Optimizer | AdamW | Weight decay giúp regularization |
| LR (encoder) | 1e-5 | Differential LR — fine-tune nhẹ backbone |
| LR (heads) | 1e-3 | Heads mới cần LR cao hơn |
| Weight decay | 1e-2 | L2 regularization |
| Max epochs | 10 | Upper bound |
| **Early stopping** | **Patience=3** | **Dừng khi val Macro-F1 không cải thiện 3 epoch liên tiếp** |
| Loss | BCEWithLogitsLoss | Binary cross-entropy với pos_weight |
| FP16 | Mixed precision | GradScaler + autocast để tiết kiệm memory |
| Gradient clipping | max_norm=1.0 | Ngăn gradient exploding |
| Gradient checkpointing | Enabled | Trade compute for memory |
| Multi-GPU | DataParallel | 2× T4 trên Kaggle |

#### 4.3.3 Đường học và điểm dừng ⚠️
- **Hình vẽ bắt buộc**: 
  - Plot train loss vs val loss theo epoch (4 axes hoặc average)
  - Plot val Macro-F1 theo epoch
- **Giải thích điểm dừng**: 
  - Early stopping patience=3: nếu val F1 không tăng sau 3 epoch → dừng & load best checkpoint
  - Lý do chọn patience=3: đủ để vượt qua local fluctuation nhưng không overfit
  - Lý do chọn val Macro-F1 làm metric theo dõi: phù hợp với imbalanced binary classification
- **Phân tích overfitting/underfitting**:
  - So sánh train loss vs val loss: nếu train giảm mà val tăng → overfitting
  - Vai trò của differential LR: tránh catastrophic forgetting trên pretrained weights

### 4.4 Baseline 3: D-DGCN (Học sâu - Graph Neural Network)

#### 4.4.1 Kiến trúc
- RoBERTa CLS → Linear(768, 256) → Dynamic Graph Construction (top-K=10 cosine similarity) → 2-layer GCN(256→128→64) → Global Mean Pool → 4 heads

#### 4.4.2 Tham số mô hình ⚠️
| Tham số | Giá trị | Giải thích |
|---------|---------|------------|
| Backbone | `roberta-base` (shared) | Chung encoder với baseline RoBERTa |
| Projection dim | 256 | Giảm chiều trước khi tạo graph |
| Top-K graph | 10 | Số cạnh/node trong dynamic graph |
| GCN hidden | 128 | Hidden dim layer GCN 1 |
| GCN output | 64 | Output dim layer GCN 2 |
| Dropout | 0.1 | Trên GCN layers |
| Training setup | Giống RoBERTa | AdamW, differential LR, early stopping patience=3 |

#### 4.4.3 Đường học và điểm dừng ⚠️
- **Hình vẽ bắt buộc**: tương tự RoBERTa — train/val loss + val F1 theo epoch
- **Giải thích**: 
  - Dynamic graph cho phép mô hình hóa quan hệ giữa các posts trong cùng batch
  - So sánh convergence speed vs RoBERTa (GCN thêm capacity nhưng cũng thêm risk overfitting)
  - Early stopping dùng cùng criteria: patience=3 trên val avg Macro-F1

### 4.5 Phương pháp đề xuất: RAG + LLM Agent

#### 4.5.1 Kiến trúc tổng thể
- **Sơ đồ kiến trúc** (vẽ): Agent = LLM + Planning + Tool Use (Function Calling) + ReAct Loop + Memory

#### 4.5.2 Thành phần RAG
| Thành phần | Chi tiết |
|------------|---------|
| Embedding model | `all-MiniLM-L6-v2` (384-dim, trên CPU) |
| Vector store | FAISS `IndexFlatIP` (cosine similarity) |
| Index 1 — Similar Posts | Training posts → FAISS → top-K few-shot examples |
| Index 2 — Knowledge Base | 4 dimension descriptions + 16 type profiles = 20 chunks |

#### 4.5.3 Thành phần Agent
| Thành phần | Chi tiết |
|------------|---------|
| LLM | Qwen2.5-7B-Instruct (4-bit NF4 quantization) |
| Function Calling | Qwen2.5 native `<tool_call>` format — 5 tools with JSON schema |
| Tools | `analyze_text`, `retrieve_similar(top_k)`, `retrieve_knowledge(query, top_k)`, `recall_experience`, `predict(type, ie, ie_conf, ...)` |
| ReAct Loop | Max 8 steps, LLM tự quyết định tool nào + tham số gì |
| Short-term Memory | Scratchpad (per-sample trace) |
| Long-term Memory | FAISS experience store (threshold=0.80), dynamic few-shot prompting |

#### 4.5.4 Tham số hệ thống ⚠️
| Tham số | Giá trị | Giải thích |
|---------|---------|------------|
| `RAG_TOP_K` | 5 | Số similar posts retrieve |
| `EMBED_DIM` | 384 | Sentence embedding dimension |
| `CONFIDENCE_THRESHOLD` | 0.55 | Ngưỡng confidence per-axis để chấp nhận prediction |
| `MAX_RETRIES` | 1 | Số lần retry nếu confidence thấp |
| `MAX_REACT_STEPS` | 8 | Giới hạn số lượt LLM reasoning |
| `MEMORY_CONF_THRESHOLD` | 0.80 | Ngưỡng lưu vào long-term memory |
| `BATCH_SIZE` | 4 | Song song hóa inference |
| LLM `temperature` | 0.1 | Gần deterministic cho consistency |
| LLM `max_new_tokens` | 150-300 | Đủ cho tool call + reasoning |

#### 4.5.5 Điểm dừng và Learning Curve ⚠️
- RAG + Agent **KHÔNG có training loop** → không có learning curve truyền thống
- **Thay thế**: Phân tích long-term memory growth qua quá trình inference:
  - Số lượng experience stored vs số sample đã xử lý
  - Trung bình confidence qua thời gian (có tăng khi memory lớn hơn?)
  - Phân bố số ReAct steps cần thiết (agent hiệu quả hơn khi có nhiều memory?)
- **Giải thích tại sao không cần "dừng học"**:
  - LLM đã được pre-train; agent chỉ thực hiện inference với RAG context
  - "Học" xảy ra qua long-term memory accumulation, không qua gradient update
  - MAX_REACT_STEPS=8 và MAX_RETRIES=1 đóng vai trò giới hạn tương tự early stopping

---

## Chương 5: Thiết lập thí nghiệm (Experimental Setup) ⚠️

### 5.1 Môi trường thực nghiệm
| Tài nguyên | Cấu hình |
|-----------|---------|
| Platform | Kaggle Notebooks |
| GPU | 2× NVIDIA T4 16GB |
| RAM | 32 GB |
| Python | 3.10+ |
| Framework | PyTorch, Transformers, FAISS |

### 5.2 Tái tạo kết quả (Reproducibility)
- `random_state=42` cho tất cả split
- Cùng tập Train/Val/Test cho tất cả 4 phương pháp
- Fixed `temperature=0.1` cho LLM

### 5.3 Metrics đánh giá
- **Per-axis** (4 bài toán nhị phân): F1-Macro, Accuracy, AUC-ROC
- **16-type**: F1-Macro, F1-Weighted, Accuracy, AUC-ROC (Macro OvR)
- **Tại sao F1-Macro**: phù hợp imbalanced data, đánh giá công bằng giữa 2 class

---

## Chương 6: Kết quả thí nghiệm (Results)

### 6.1 Kết quả trên 4 trục nhị phân
- **Bảng so sánh chính**: 4 phương pháp × 4 axes × 3 metrics
  
| Axis | Metric | SVM | RoBERTa | D-DGCN | RAG+Agent |
|------|--------|-----|---------|--------|-----------|
| I/E | F1-Macro | ... | ... | ... | ... |
| S/N | F1-Macro | ... | ... | ... | ... |
| T/F | F1-Macro | ... | ... | ... | ... |
| J/P | F1-Macro | ... | ... | ... | ... |
| **Avg** | **F1-Macro** | ... | ... | ... | ... |

### 6.2 Kết quả trên 16 types
- Accuracy, F1-Macro, F1-Weighted, AUC-ROC cho mỗi phương pháp
- Per-type classification report (precision/recall/F1 cho từng type)

### 6.3 Đường học của mô hình học sâu ⚠️
- **Hình 6.3a**: RoBERTa — Train/Val Loss theo Epoch
- **Hình 6.3b**: RoBERTa — Val Macro-F1 theo Epoch (đánh dấu best epoch & early stop point)
- **Hình 6.3c**: D-DGCN — Train/Val Loss theo Epoch
- **Hình 6.3d**: D-DGCN — Val Macro-F1 theo Epoch
- **Phân tích**: 
  - Epoch nào đạt best val F1? Tại sao dừng tại đó?
  - Dấu hiệu overfitting (nếu có)?
  - So sánh tốc độ convergence giữa RoBERTa và D-DGCN

### 6.4 Phân tích Agent Behavior
- Phân bố tool usage: % samples gọi mỗi tool
- Trung bình số ReAct steps per sample
- Tỉ lệ fallback predictions
- Long-term memory: số experience stored, confidence distribution
- **Demo trace** cho 2-3 samples minh họa

### 6.5 Phân tích thời gian chạy
| Phương pháp | Training time | Inference time (per sample) |
|------------|--------------|---------------------------|
| SVM | ~minutes | <0.01s |
| RoBERTa | ~X min (Y epochs) | ~0.1s |
| D-DGCN | ~X min (Y epochs) | ~0.15s |
| RAG+Agent | 0 (no training) | ~25-30s (batched) |

---

## Chương 7: Thảo luận (Discussion)

### 7.1 So sánh tổng thể
- Phương pháp nào tốt nhất cho từng axis? Tại sao?
- Trade-off: accuracy vs inference speed vs interpretability

### 7.2 Phân tích ưu/nhược điểm từng phương pháp
| | SVM | RoBERTa | D-DGCN | RAG+Agent |
|--|-----|---------|--------|-----------|
| **Ưu** | Nhanh, đơn giản | Contextual embeddings | Graph relations | Không cần training, interpretable, có memory |
| **Nhược** | Mất ngữ cảnh | Cần fine-tune | Complex, slow train | Chậm inference, phụ thuộc LLM quality |

### 7.3 Phân tích theo chiều MBTI
- Chiều nào dễ phân loại nhất? (thường I/E)
- Chiều nào khó nhất? (thường T/F hoặc J/P)
- Giải thích từ góc độ ngôn ngữ học

### 7.4 Vai trò của RAG và Agent components
- Ablation ngầm: so sánh khi có/không có retrieve_knowledge, recall_experience
- Knowledge base có thực sự giúp LLM phân loại tốt hơn?

### 7.5 Hạn chế
- Dataset bias (PersonalityCafe — self-reported, may be inaccurate)
- Class imbalance nghiêm trọng (INFP, INFJ chiếm đa số)
- LLM inference cost cao

---

## Chương 8: Kết luận và Hướng phát triển

### 8.1 Kết luận
- Tóm tắt kết quả chính
- Phương pháp nào phù hợp nhất cho bài toán MBTI?

### 8.2 Hướng phát triển
- Fine-tune LLM (LoRA/QLoRA) thay vì chỉ dùng inference
- Multi-Agent system cho từng trục MBTI
- Larger LLM (14B, 72B) hoặc API-based (GPT-4o)
- Cross-lingual MBTI detection

---

## Phụ lục

### A. Chi tiết Knowledge Base
- Bảng đầy đủ MBTI_DIM_KNOWLEDGE (4 chiều)
- Bảng đầy đủ MBTI_TYPE_PROFILES (16 types với cognitive functions)

### B. Ví dụ Agent Trace
- 2-3 ví dụ đầy đủ: posts → tool calls (với arguments) → observations → prediction
- Minh họa tính tự trị: LLM tự chọn tool, tự viết query cho retrieve_knowledge

### C. Confusion Matrices
- 4 confusion matrices (per axis) cho mỗi phương pháp
- 16×16 confusion matrix cho full type classification

### D. Mã nguồn
- Link notebook Kaggle hoặc GitHub

---

## Checklist theo yêu cầu giảng viên ⚠️

| Yêu cầu | Vị trí trong báo cáo |
|---------|---------------------|
| Tham số mô hình | Bảng tham số tại §4.2.2, §4.3.2, §4.4.2, §4.5.4 |
| Thiết lập thí nghiệm | Chương 5 (môi trường, metrics, reproducibility) |
| Đường học mô hình | §6.3 (RoBERTa & D-DGCN train/val curves) |
| Tại sao dừng học tại điểm đó | §4.3.3, §4.4.3 (early stopping analysis), §4.5.5 (RAG+Agent: no training) |
| Mô tả rõ đường học | §6.3 Hình 6.3a-d + phân tích overfitting/convergence |
