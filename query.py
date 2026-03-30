import base64
import requests

# ------------------------------------------------
# Config
# ------------------------------------------------
oai_config = {
    'apikey': 'sk-d2e3e0a96dd941eb92555a105a93eab9',
    'apibase': "https://dashscope.aliyuncs.com/compatible-mode/v1",
    'model': "qwen3.5-flash",
    'api_mode': "chat_completions",
    'max_retries': 2,
    'connect_timeout': 10,
    'read_timeout': 120
}

def encode_image(image_path: str):

    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def qwen_vl_generate_query(image_path: str, question: str):

    image_b64 = encode_image(image_path)

    url = f"{oai_config['apibase']}/chat/completions"

    headers = {
        "Authorization": f"Bearer {oai_config['apikey']}",
        "Content-Type": "application/json"
    }

    prompt = """
You are a search query generator for a multimodal RAG system.

Given an image and a question, generate a concise web search query \\
that helps retrieve information needed to answer the question.

Rules:
- return ONLY the search query
- keep it short (3 to 10 words)
- include key entities if possible
"""

    payload = {
        "model": oai_config["model"],
        "messages": [
            {
                "role": "system",
                "content": prompt
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": question
                    }
                ]
            }
        ],
        "temperature": 0.2
    }

    resp = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=(oai_config["connect_timeout"], oai_config["read_timeout"])
    )

    resp.raise_for_status()

    data = resp.json()

    query = data["choices"][0]["message"]["content"]

    return query.strip()

def image_to_query(image_path: str, question: str):
    resposne = qwen_vl_generate_query(image_path=image_path,
                                      question=question)
    return resposne
    

def main():
    resposne = qwen_vl_generate_query(image_path="/remote-home1/xzhe/projects/CV_project/RAG_identifier/data/inference_query/BigBen_b.jpg",
                                      question="Where is this building?")
    print(f"Web query: \"{resposne}\"")
    
if __name__ == "__main__":
    main()