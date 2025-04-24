from transformers import pipeline

# Initialize the zero-shot classification pipeline
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Define the candidate labels
labels = ["Bug", "Product Question", "Feature Request", "Sales", "Spam"]

# Define the few-shot examples
few_shot_prompt = """
Classify the following messages into one of the categories: Bug, Product Question, Feature Request, Sales, Spam.

Example 1:
Message: "Hi! What is your pricing? Can I talk with someone?"
Classification: Sales

Example 2:
Message: "The app crashes when I try to upload a photo."
Classification: Bug

Example 3:
Message: "Can you add a dark mode feature?"
Classification: Feature Request

Example 4:
Message: "Hi, how much is Atlas?"
Classification: Product Question

Example 5:
Message: "Buy now and get 50% off!"
Classification: Spam
"""


# List of messages to classify
messages = [
    "Hi! What is your pricing? Can I talk with someone?",
    "Hi! Yes, we'd love to hop on a phone call! When are you free to chat?",
    "Hi, how much is Atlas?",
    "Looks like it, yeah.",
    "hey not super urgent but fyi in",
]

# Classify messages using the zero-shot classification model with few-shot context
for message in messages:
    result = classifier(few_shot_prompt, candidate_labels=labels)
    classification = result["labels"][0]
    print(f"Message: {message}")
    print(f"Classification: {classification}\n")

