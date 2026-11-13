# Evaluation report

## Classifier (PyTorch + HuggingFace)

- champion version: 2  (evaluated on 1800 test rows)
- accuracy: **0.8044**
- macro F1: **0.8042**

| product group | accuracy |
|---|---|
| Checking or savings account | 0.785 |
| Credit card | 0.71 |
| Credit reporting | 0.83 |
| Debt collection | 0.775 |
| Money transfer or service | 0.775 |
| Mortgage | 0.86 |
| Payday or personal loan | 0.695 |
| Student loan | 0.955 |
| Vehicle loan or lease | 0.855 |

## Retrieval (FAISS over Regulation E)

- recall@5: **0.84** on 25 questions

Misses:

- A merchant debits a different amount each month. Must I get advance notice of the amount?  expected ['1005.10'], got ['1005.8', '1005.17', '1005.17', '1005.14', '1005.11']
- Can a lender require that I repay a loan by automatic electronic debit?  expected ['1005.10'], got ['1005.17', '1005.18', '1005.18', '1005.18', '1005.17']
- Is a bank required to disclose its ATM fee on the screen or a sign before I complete a withdrawal?  expected ['1005.16'], got ['1005.7', '1005.18', '1005.17', '1005.18', '1005.17']
- Can a bank send me a debit card I did not request?  expected ['1005.5'], got ['1005.17', '1005.14', '1005.17', '1005.14', '1005.18']

## Relief model (TensorFlow)

- champion version: 1
- test PR-AUC: **0.2982**
- test ROC-AUC: 0.8764
- precision / recall @ threshold: 0.176 / 0.800
