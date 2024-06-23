
# Sortify

## Instructions
- Install Poetry: `python3 -m pip install poetry`
- Install dependencies: `poetry install`
- Download spaCy model: `python3 -m spacy download en_core_web_sm`

## 1. Standard ML Approach with SpaCy, Scikit-learn, and Pandas

I experimented with 3 different models for training (details in `ml_train.ipynb`):

### Experiment 1: `data_preprocessing_v1.ipynb`
- Data cleaning
- Duplicate removal
- Outlier removal
- Handling missing values
- Test set accuracy: ~40%

### Experiment 2: `data_preprocessing_v2.ipynb`
- Similar to v1, with additional tag filtering
- Test set accuracy: ~40%

### Experiment 3: `data_preprocessing_v3.ipynb`
- Grouped by `ticket_number`, sorted by `message_created_on`
- Selected first message from each group
- Test set accuracy: ~30%

### Experiment 4: `data_preprocessing_v4.ipynb`
- Grouped and concatenated all messages per ticket
- Test set accuracy: ~50%

**Challenges:** Limited dataset size (~1000 entries).

#### Next Steps:
- Increase training data volume
- Conduct deeper analysis
- Explore alternative models
- Review preprocessing steps
- Consider neural network models
- Perform hyperparameter tuning

## 2. LLM Approach with Hugging Face

No preprocessing was applied; using raw ticket data:

### Hugging Face Models
- Accuracy: ~40%

### OpenAI GPT-3.5 Model
- Accuracy: ~80%

#### Next Steps:
- Perform feature engineering
- Optimize prompt tuning
- Consider advanced LLMs like GPT-4 or LAMA

Files: `zero_shot_classification.py`, `few_shots_classification_openai.py`, `few_shots_classification_hf.py`

## 3. NN approach

Tried with preprocessed and raw data.

### Transfer learning 
- Accuracy: ~50%

#### Next Steps:
- Optimize hyperparameters
- The size after preprocessing is too small for deep learning (now we have~1000 entries)
- Maybe something off with data

Files: `nn_classification.ipynb`

## Summary

To address the challenge of sorting tickets with various complexities such as different languages, missing values, and overlapping high-frequency words, we can consider both machine learning (ML) and large language model (LLM) approaches.

#### Data Challenges

Using Python for fast text analysis, I identified that several models could potentially be used for ticket sorting. However, the dataset presents several challenges:
- **Multilingual Data:** The dataset contains tickets in different languages.
- **Missing Values:** Some tickets have incomplete information.
- **High-Frequency Words:** There are words that appear frequently across different categories, causing overlap.

#### Decision Factors

- **Budget:** The LLM approach might be more cost-effective if the translation cost in the ML approach is high.
- **Data Size:** For a larger dataset, the LLM approach could be more scalable and require less preprocessing.
- **Performance:** LLMs can handle more complexity and variability in the data, potentially leading to better performance with less effort in preprocessing.

## Conclusion

Both ML and LLM approaches have their advantages and trade-offs. The final decision should consider the specific constraints of budget, data size, and required performance.
