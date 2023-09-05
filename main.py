from fastapi import FastAPI
from pydantic import BaseModel
import requests
import uuid
import time
import config

app = FastAPI()

class Item(BaseModel):
    imgUrl: str

# 테스트용 메서드
@app.get("/")
async def test():
    response = 'GET 메서드 실행 완료'
    return {response}

# 클로바 OCR
@app.post("/ocr")
async def send_request(item: Item):
    url = config.OCR_URL
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
    ocrResponse = requests.post(url, headers=headers, json=data)

    result = ' '.join([field['inferText'] for image in ocrResponse.json()['images'] for field in image['fields']])

    response = {"result": result}
    return response


# FastAPI 앱 실행
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
