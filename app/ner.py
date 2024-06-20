import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")


def extract_names(text):
    client = OpenAI(api_key=openai_api_key)
    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that extracts names and last names from text.",
            },
            {
                "role": "user",
                "content": f"Extract only the names and last names from the following text and return them as a comma-separated list without any additional text or explanations: {text}",
            },
        ],
        model="gpt-4o",  # "gpt-3.5-turbo",  # or "gpt-4"
        max_tokens=100,
        temperature=1.0,
    )
    # Access the response content correctly
    answer = response.choices[0].message.content.strip()
    return answer
