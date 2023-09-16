import time
import uuid
import requests
import openai
from config import *

def extract_full_content_with_ocr(item):
    headers = {
        'X-OCR-SECRET': X_OCR_SECRET,
        'Content-Type': 'application/json',
    }
    data = {
        "version": "V1",
        "requestId": str(uuid.uuid4()),
        "timestamp": int(round(time.time() * 1000)),
        "lang": "ko",
        "images": [
            {
                "format": "jpg",
                "name": "demo",
                "url": item.imgUrl
            }
        ],
    }
    response = requests.post(OCR_URL, headers=headers, json=data)
    response_data = response.json()

    if response.status_code != 200 or 'images' not in response_data:
        raise Exception("Failed to process OCR request.")

    return ' '.join([field['inferText'] for image in response_data['images'] for field in image['fields']])

def extract_foods_with_gpt(txt):
    openai.organization = OPEN_AI_ORGANIZATION
    openai.api_key = OPENAI_API_KEY
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{
            "role": "user",
            "content": txt + " 라는 텍스트 중에서, 식품의 원재료만 추출해서 줘라. 구분은 쉼표로 해줘."
        }]
    )

    if 'choices' not in response or not response['choices']:
        raise Exception("Failed to process GPT request.")

    return response['choices'][0]['message']['content']

