
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
- Test set accuracy: ~40%

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
- Accuracy: ~60%

#### Next Steps:
- Perform feature engineering
- Optimize prompt tuning
- Consider advanced LLMs like GPT-4 or LAMA

Files: `zero_shot_classification.py`, `few_shots_classification_openai.py`, `few_shots_classification_hf.py`