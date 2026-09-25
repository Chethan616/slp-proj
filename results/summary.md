# Results

## Model comparison (1500 held-out test utterances, 41 classes)

| Model | Family | Trainable parameters | Test accuracy | Macro F1 | Training time |
|---|---|---:|---:|---:|---:|
| TF-IDF + Logistic Regression | classical | 112,864 | 0.8750 | 0.8803 | 1s |
| Bag-of-words + MLP | neural (from scratch) | 249,744 | 0.8867 | 0.8847 | 5s |
| Embedding + BiLSTM | neural (from scratch) | 209,040 | 0.8867 | 0.8763 | 27s |
| DistilBERT (fine-tuned) **(deployed)** | transformer (pre-trained) | 66,965,776 | 0.9333 | 0.9294 | 887s |

## Out-of-scope handling (deployed model)

Threshold **0.38**, selected by maximising macro F1 on the validation split and reported below on test.

| Split | In-scope accuracy | Out-of-scope recall | Macro F1 |
|---|---:|---:|---:|
| Test, no threshold | 0.9356 | 0.9267 | 0.9294 |
| Validation, threshold 0.38 | 0.9333 | 0.9467 | 0.9390 |
| Test, threshold 0.38 | 0.9222 | 0.9467 | 0.9273 |

## Most frequent confusions (deployed model)

| True intent | Predicted as | Count |
|---|---|---:|
| oos | recipe | 4 |
| restaurant_reviews | meal_suggestion | 3 |
| recipe | ingredients_list | 3 |
| oos | restaurant_suggestion | 3 |
| ingredients_list | recipe | 3 |
| accept_reservations | restaurant_reservation | 3 |
| restaurant_suggestion | meal_suggestion | 2 |
| restaurant_reviews | oos | 2 |
| restaurant_reservation | accept_reservations | 2 |
| oos | cook_time | 2 |

## Hardest intents by F1 (deployed model)

| Intent | F1 | Support |
|---|---:|---:|
| recipe | 0.800 | 30 |
| meal_suggestion | 0.839 | 30 |
| restaurant_reservation | 0.871 | 30 |
| restaurant_reviews | 0.889 | 30 |
| ingredients_list | 0.900 | 30 |
| restaurant_suggestion | 0.903 | 30 |
| accept_reservations | 0.915 | 30 |
| how_busy | 0.915 | 30 |
| cook_time | 0.935 | 30 |
| oos | 0.952 | 150 |
