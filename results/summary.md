# Results

## Model comparison (1500 held-out test utterances, 41 classes)

| Model | Family | Trainable parameters | Test accuracy | Macro F1 | Training time |
|---|---|---:|---:|---:|---:|
| TF-IDF + Logistic Regression | classical | 485,071 | 0.8800 | 0.9055 | 3s |
| Bag-of-words + MLP | neural (from scratch) | 375,081 | 0.8747 | 0.8981 | 5s |
| Embedding + BiLSTM | neural (from scratch) | 273,833 | 0.8860 | 0.9015 | 39s |
| DistilBERT (fine-tuned) **(deployed)** | transformer (pre-trained) | 66,985,001 | 0.9460 | 0.9556 | 1297s |

## Out-of-scope handling (deployed model)

Threshold **0.44**, selected by maximising macro F1 on the validation split and reported below on test.

| Split | In-scope accuracy | Out-of-scope recall | Macro F1 |
|---|---:|---:|---:|
| Test, no threshold | 0.9742 | 0.8333 | 0.9556 |
| Validation, threshold 0.44 | 0.9663 | 0.9100 | 0.9672 |
| Test, threshold 0.44 | 0.9700 | 0.9033 | 0.9651 |

## Most frequent confusions (deployed model)

| True intent | Predicted as | Count |
|---|---|---:|
| oos | shopping_list | 6 |
| oos | directions | 5 |
| oos | recipe | 4 |
| oos | pto_balance | 4 |
| oos | pay_bill | 4 |
| oos | balance | 4 |
| oos | weather | 3 |
| distance | directions | 3 |
| transactions | todo_list | 2 |
| shopping_list | todo_list | 2 |

## Hardest intents by F1 (deployed model)

| Intent | F1 | Support |
|---|---:|---:|
| distance | 0.847 | 30 |
| transactions | 0.847 | 30 |
| shopping_list | 0.857 | 30 |
| directions | 0.870 | 30 |
| balance | 0.879 | 30 |
| oos | 0.901 | 300 |
| pto_balance | 0.903 | 30 |
| todo_list | 0.909 | 30 |
| pay_bill | 0.923 | 30 |
| time | 0.933 | 30 |
