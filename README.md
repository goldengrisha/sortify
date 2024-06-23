# Sortify

## Setup Instructions

1. **Install Poetry:**
   ```sh
   python3 -m pip install poetry
   ```
2. **Install dependencies:**
   ```sh
   poetry install
   ```
3. **Download spaCy model:**
   ```sh
   python3 -m spacy download en_core_web_sm
   ```

## 1. Standard ML Approach with SpaCy, Scikit-learn, and Pandas

I experimented with three different models for training, detailed in `ml_train.ipynb`:

### Experiment 1: `data_preprocessing_v1.ipynb`
- Data cleaning
- Duplicate removal
- Outlier removal
- Handling missing values
- Test set accuracy: ~40%

### Experiment 2: `data_preprocessing_v2.ipynb`
- Similar to v1, with additional tag filtering
- Test set accuracy: ~50%

### Experiment 3: `data_preprocessing_v3.ipynb`
- Grouped by `ticket_number`, sorted by `message_created_on`
- Selected first message from each group
- Test set accuracy: ~40%

### Experiment 4: `data_preprocessing_v4.ipynb`
- Grouped and concatenated all messages per ticket
- Test set accuracy: ~60%

**Challenges:** Limited dataset size (~1000 entries).

#### Next Steps:
- Increase training data volume
- Conduct deeper analysis
- Explore alternative models
- Review preprocessing steps
- Consider neural network models
- Perform hyperparameter tuning

## 2. LLM Approach with Hugging Face

Using raw ticket data without preprocessing:

### Hugging Face Models
- Accuracy: ~40%

### OpenAI GPT-3.5 Model
- Accuracy: ~80%

#### Next Steps:
- Perform feature engineering
- Optimize prompt tuning
- Consider advanced LLMs like GPT-4 or LLAMA

Files: `zero_shot_classification.py`, `few_shots_classification_openai.py`, `few_shots_classification_hf.py`

## 3. Neural Network Approach

Experimented with both preprocessed and raw data.

### Transfer Learning 
- Accuracy: ~60%

#### Next Steps:
- Optimize hyperparameters
- Address the small dataset size (currently ~1000 entries)
- Investigate potential data issues

Files: `nn_classification.ipynb`

## Summary

The classes are somewhat imbalanced, particularly the "Spam" category, which has significantly fewer entries compared to the other classes. This imbalance can affect the classifier's performance.

### Data Challenges

Using Python for fast text analysis, I identified several challenges with the dataset:
- **Multilingual Data:** The dataset contains tickets in different languages.
- **Missing Values:** Some tickets have incomplete information.
- **High-Frequency Words:** Common words appear frequently across different categories, causing overlap.

### Decision Factors

- **Budget:** The LLM approach might be more cost-effective if the translation cost in the ML approach is high.
- **Data Size:** For a larger dataset, the LLM approach could be more scalable and require less preprocessing.
- **Performance:** LLMs can handle more complexity and variability in the data, potentially leading to better performance with less effort in preprocessing.

## Conclusion

Both ML and LLM approaches have their advantages and trade-offs. The final decision should consider the specific constraints of budget, data size, and required performance.
