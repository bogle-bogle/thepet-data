# -*- coding: utf-8 -*-
import time
import uuid

import requests
from fastapi import FastAPI
from pydantic import BaseModel
import openai

import config

app = FastAPI()


class ImgItem(BaseModel):
    imgUrl: str

class TxtItem(BaseModel):
    content: str

# 테스트용 메서드
@app.get("/")
async def test():
    response = 'GET 메서드 실행 완료'
    return {response}


# 클로바 OCR
@app.post("/ocr")
async def send_request_to_clova(item: ImgItem):
    result = extract_full_content_with_ocr(item)
    return {"content": result}


@app.post("/gpt")
async def send_request_to_gpt(item: TxtItem):
    return extract_foods_with_gpt(item)

@app.post("/extract")
async def extract_foods_from_img(item: ImgItem):
    ocrResult = extract_full_content_with_ocr(item)
    gptResult = extract_foods_with_gpt({"content": ocrResult})
    print(gptResult) 
    return gptResult

def extract_full_content_with_ocr(item : ImgItem):
    headers = {
        'X-OCR-SECRET': config.X_OCR_SECRET,
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
    ocrResponse = requests.post(config.OCR_URL, headers=headers, json=data)

    return ' '.join([field['inferText'] for image in ocrResponse.json()['images'] for field in image['fields']])

def extract_foods_with_gpt(item: TxtItem):
    openai.organization = config.OPEN_AI_ORGANIZATION
    openai.api_key = config.OPENAI_API_KEY
    openai.Model.list()

    gptResponse = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": item['content'] + " 라는 텍스트 중에서, 식품의 원재료만 추출해서 줘라. 구분은 쉼표로 해줘."}]
    )
    print(gptResponse)
    return gptResponse

# FastAPI 앱 실행
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)