import requests

from pathlib import Path

IDENTIFIER_DIR = Path(__file__).resolve().parent
PROJECT_BASE_DIR = IDENTIFIER_DIR.parent

url = "http://127.0.0.1:18000/predict"

files = {
    "file": open(IDENTIFIER_DIR / "data" / "inference_query" / "Sacred_Heart_Cathedral_of_Guangzhou_2025.06_02.jpg", 
                 "rb")
}

data = {
    "question": "Where is this building?"
}

import time


t1 = time.time()

response = requests.post(url, files=files, data=data)

t2 = time.time()

print(f"Time per inference: {t2 - t1}")
# print("status:", response.status_code)
# print("headers:", response.headers)
print("text:", response.text)