from openai import OpenAI


open_ai_api_key = ""

if open_ai_api_key == "":
    raise ValueError("Please provide your OpenAI API key.")


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

client = OpenAI(api_key=open_ai_api_key)

# Classify messages using OpenAI's GPT-3
for message in messages:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": few_shot_prompt},
            {"role": "user", "content": message},
        ],
    )
    if response["choices"][0]["finish_reason"] != "stop":
        raise Exception("Unexpected error during classification")

    classification = response["choices"][0]["message"]["content"].strip()

    print(f"Message: {message}")
    print(f"Classification: {classification}\n")
