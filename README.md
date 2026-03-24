#cứ thế mà chạy thôiii

##đây là output 
```bash
Loading dataset …
  Total users : 8675

Splitting data (60 / 20 / 20) …
  Train: 5205  Val: 1735  Test: 1735

Building feature matrix (word-TF-IDF + char-TF-IDF + lexicon) …
  Feature shape: (5205, 25008)

Training classifiers …
  [IE] SMOTE: 5205 → 8006 samples
  [NS] Searching best C on val set …
    C=0.01    val macro-F1=0.6221
    C=0.05    val macro-F1=0.6359
    C=0.1     val macro-F1=0.6534
    C=0.5     val macro-F1=0.6856
    C=1.0     val macro-F1=0.6931
    C=2.0     val macro-F1=0.6946
    C=5.0     val macro-F1=0.6961
  [NS] Best C=5.0  val macro-F1=0.6961
  [NS] SMOTE: 5205 → 8970 samples
  [NS] Best threshold=0.55  val macro-F1=0.7070
  Thresholds: {'IE': 0.5, 'NS': 0.5500000000000002, 'TF': 0.5, 'JP': 0.5}

Building knowledge base from training data …
==============================================================
DETAILED CLASSIFICATION REPORT
==============================================================

=== IE  (I/E)  thr=0.50  macro-F1=0.693 ===
              precision    recall  f1-score   support

           I       0.52      0.53      0.52       382
           E       0.87      0.86      0.86      1353

    accuracy                           0.79      1735
   macro avg       0.69      0.69      0.69      1735
weighted avg       0.79      0.79      0.79      1735


=== NS  (N/S)  thr=0.55  macro-F1=0.686 ===
              precision    recall  f1-score   support

           N       0.51      0.41      0.45       246
           S       0.91      0.93      0.92      1489

    accuracy                           0.86      1735
   macro avg       0.71      0.67      0.69      1735
weighted avg       0.85      0.86      0.85      1735


=== TF  (T/F)  thr=0.50  macro-F1=0.819 ===
              precision    recall  f1-score   support

           T       0.85      0.82      0.83       937
           F       0.79      0.82      0.81       798

    accuracy                           0.82      1735
   macro avg       0.82      0.82      0.82      1735
weighted avg       0.82      0.82      0.82      1735


=== JP  (J/P)  thr=0.50  macro-F1=0.715 ===
              precision    recall  f1-score   support

           J       0.79      0.76      0.77      1066
           P       0.64      0.67      0.66       669

    accuracy                           0.73      1735
   macro avg       0.71      0.72      0.71      1735
weighted avg       0.73      0.73      0.73      1735


========================================================================
PAPER-STYLE EVALUATION TABLE  (ref: arXiv:2509.04461)
Metrics: Accuracy | macro-F1 | Clf-AUC | RAG-AUC   [top_k=5]
========================================================================
  Computing RAG label fractions for 1735 test users × 4 axes …

Axis       Acc  macro-F1   Clf-AUC   RAG-AUC    thr
---------------------------------------------------
IE      0.7879    0.6935    0.7975    0.5758   0.50   (I/E)
NS      0.8599    0.6856    0.8187    0.6080   0.55   (N/S)
TF      0.8202    0.8195    0.8939    0.7023   0.50   (T/F)
JP      0.7274    0.7150    0.7855    0.5851   0.50   (J/P)
---------------------------------------------------
Avg     0.7988    0.7284    0.8239    0.6178

========================================================================
16-type exact match (clf-only) : 0.4380  (43.8%)
========================================================================

Assembling orchestrator …

  Running full pipeline on 1735 test users (top_k=5) …
    200/1735 …
    400/1735 …
    600/1735 …
    800/1735 …
    1000/1735 …
    1200/1735 …
    1400/1735 …
    1600/1735 …

==============================================================
FULL PIPELINE EVALUATION  (clf + RAG + judge + threshold)
==============================================================
  16-type exact match : 0.4271  (42.7%)
  Mean confidence     : 0.2406
  Per-axis accuracy   : {'IE': '0.8023', 'NS': '0.8674', 'TF': '0.8075', 'JP': '0.7222'}
==============================================================

==============================================================
DEMO — 3 test users  (top_k=5)
==============================================================

[User #0]  True: INTP  Predicted: INTP  ✓
  Confidence : 0.2825
  Per-axis   : {'IE': 0.1872, 'NS': 0.5912, 'TF': 0.2147, 'JP': 0.1368}
  [IE] model=0.686  rag_frac=1.00  fused=0.749  thr=0.50  → I
    [INFP] score=0.175  "the bolded part is important if the other person is not willing then you re just wasting your time a"
    [ISTP] score=0.167  "for me its knowing that i m good at a lot of things so picking a career and sticking to it really ea"
  [NS] model=0.948  rag_frac=0.80  fused=0.918  thr=0.55  → N
    [INFP] score=0.166  "that s an opinion not a fact that being said i believe in the inherent right of everyone to be as bi"
    [INTP] score=0.162  "once you realize you are an you either will become more intpish or instead try to evolve and be more"
  [TF] model=0.857  rag_frac=0.40  fused=0.766  thr=0.50  → T
    [INTJ] score=0.164  "i don t really have much space for a guilty pleasure usually a guilty pleasure would require that yo"
    [ENFP] score=0.161  "not gonna happen genetically inferior is a value judgment why the fuck should i think about that obj"
  [JP] model=0.307  rag_frac=0.20  fused=0.286  thr=0.50  → P
    [INFP] score=0.161  "a healer huh? to be honest i ve always tried to fight that archetype i d rather not be a healer but "
    [ISFP] score=0.155  "my long time best friend is self typed as an as far as i know it s accurate also i m very close to m"

[User #1]  True: INTJ  Predicted: ENTJ  ✗
  Confidence : 0.1560
  Per-axis   : {'IE': 0.0057, 'NS': 0.5127, 'TF': 0.0577, 'JP': 0.0477}
  [IE] model=0.369  rag_frac=0.80  fused=0.455  thr=0.50  → E
    [ISTJ] score=0.187  "her and i have known each other for over years we dated a bit last year never got exclusive serious "
    [INFJ] score=0.186  "do you think that there should be sparks on a first date? if that s not a requirement how many dates"
  [NS] model=0.868  rag_frac=1.00  fused=0.894  thr=0.55  → N
    [INTP] score=0.185  "i am dreading my birthday i feel all this pressure to make sure i m happy and feel special and right"
    [INFJ] score=0.180  "do you think that there should be sparks on a first date? if that s not a requirement how many dates"
  [TF] model=0.751  rag_frac=0.20  fused=0.641  thr=0.50  → T
    [INTP] score=0.187  "i am dreading my birthday i feel all this pressure to make sure i m happy and feel special and right"
    [INFJ] score=0.183  "do you think that there should be sparks on a first date? if that s not a requirement how many dates"
  [JP] model=0.635  rag_frac=0.60  fused=0.628  thr=0.50  → J
    [ENFP] score=0.189  "are there any s that are incredibly quiet when you first meet someone or a group? this makes me ques"
    [ENTJ] score=0.186  "whew i almost lost more faith in humanity i will take this heavily into consideration thank you just"

[User #2]  True: INTP  Predicted: INTP  ✓
  Confidence : 0.4517
  Per-axis   : {'IE': 0.1986, 'NS': 0.9461, 'TF': 0.6255, 'JP': 0.0364}
  [IE] model=0.695  rag_frac=1.00  fused=0.756  thr=0.50  → I
    [INTP] score=0.164  "as we have many desktops called activities here at linux kde the geek s wet dream standard developme"
    [INFJ] score=0.163  "thanks for taking this poll and sharing it what you found does not surprise me but i don t think the"
  [NS] model=0.992  rag_frac=1.00  fused=0.994  thr=0.55  → N
    [INTP] score=0.175  "as we have many desktops called activities here at linux kde the geek s wet dream standard developme"
    [INTP] score=0.173  "you can say that about a great number of diseases on record i honestly don t know how to show you ho"
  [TF] model=0.910  rag_frac=1.00  fused=0.928  thr=0.50  → T
    [INTP] score=0.183  "as we have many desktops called activities here at linux kde the geek s wet dream standard developme"
    [INTP] score=0.166  "yea that s right i didn t think so ok but you need to understand that rulers would equate every man "
  [JP] model=0.485  rag_frac=0.00  fused=0.388  thr=0.50  → P
    [INTP] score=0.165  "as we have many desktops called activities here at linux kde the geek s wet dream standard developme"
    [INTP] score=0.159  "yea that s right i didn t think so ok but you need to understand that rulers would equate every man "
```