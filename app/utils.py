import time
import uuid
import requests
import openai
from config import *
import re


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

    extracted_text = ' '.join(
        [field['inferText'] for image in response_data['images'] for field in image['fields']])

    # 다양한 시작 패턴 리스트
    start_patterns = ["사용한 원료의 명칭", "원료", "사용원료", "사용한 원료", "등록성분 주원료", "원재료명 및 함량", "제조원료", "사용한 원료의 명칭:", "원료:",
                      "사용원료:", "등록성분 주원료:", "원재료명 및 함량:", "제조원료:", "(사용한 원료의 명칭)", "(원료)", "(사용원료)", "(등록성분 주원료)", "(원재료명 및 함량)", "(제조원료)"]

    ingredient_text = ""
    start_index = -1
    print(extracted_text)
    # 여러 시작 패턴을 반복하여 검색
    for pattern in start_patterns:
        start_index = extracted_text.find(pattern)
        if start_index != -1:
            start_index += len(pattern)
            ingredient_text = extracted_text[start_index:]
            break

    # 시작 패턴을 찾지 못한 경우 처리
    if start_index == -1:
        print("시작 패턴을 찾을 수 없습니다.")
        ingredient_text = extracted_text

    input_string = ingredient_text.replace(" ", "")
    input_string = re.sub(r'\([^)]*\)', '', input_string)
    input_string = re.sub(r'[^\w\s,]', '', input_string)
    input_string = re.sub(r'\s+', ' ', input_string)  # 중복 공백을 단일 공백으로 바꿈
    input_string = re.sub(r'\s,', ',', input_string)  # 띄어쓰기와 쉼표를 쉼표로 대체
    input_string = input_string.replace(",", ", ")
    print(input_string)
    print(input_string.replace(", ", ",").split(","))

    return input_string


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
