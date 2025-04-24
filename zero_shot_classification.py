from transformers import pipeline

# Initialize the text classification pipeline
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Define the candidate labels
labels = ["Bug", "Product Question", "Feature Request", "Sales", "Spam"]

# List of messages to classify
messages = [
    "Hi! What is your pricing? Can I talk with someone?",
    "Hi! Yes, we'd love to hop on a phone call! When are you free to chat?",
    "Hi, how much is Atlas?",
    "Looks like it, yeah.",
    "hey not super urgent but fyi in",
]

# Classify messages
for message in messages:
    result = classifier(message, candidate_labels=labels)
    print(f"Message: {message}")
    print(f"Classification: {result['labels'][0]} (Score: {result['scores'][0]:.4f})\n")
