import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Khởi tạo client với endpoint của Azure và token của GitHub
client = OpenAI(
    base_url="https://models.inference.ai.azure.com",
    api_key=os.getenv("GITHUB_TOKEN", ""),
)

if __name__ == "__main__":
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Xin chào, bạn là ai?"}],
    )
    print(response.choices[0].message.content)