import asyncio
import os
import base64
from pathlib import Path
from mistralai import Mistral

image_path = "test_24915.jpg"
temperature = 0.0
max_tokens = None

def image_to_base64(image_path):
    """Encode the image to base64."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"Error: The file {image_path} was not found.")
        return None
    except Exception as e:  # Added general exception handling
        print(f"Error: {e}")
        return None


client = Mistral(api_key = os.environ["MISTRAL_API_KEY"])

question = """
reply to the question based on the document. Reply with the answers only, don't add a period at the end of the answer.
question: Please write out the expression of the formula in the image using LaTeX format.
answer:
"""


def predict(image_path: str, question: list[str]) -> str:
    image_base64 = image_to_base64(image_path)
    base64_messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": question,
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}",
                    },
                },
            ],
        }
    ]

    response = client.chat.complete(
        model="pixtral-12b-2409",
        messages=base64_messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return {"extracted_answer": response.choices[0].message.content}

print("="*100)    
res = predict(image_path, question)
print(res)