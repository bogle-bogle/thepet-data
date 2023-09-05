# -*- coding: utf-8 -*-
import time
import uuid

import requests
from fastapi import FastAPI
from pydantic import BaseModel
import openai

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
async def send_request_to_clova(item: Item):
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


@app.get("/gpt")
async def send_request_to_gpt():
    openai.organization = "org-OjnIeB7qK2AAoKPCxrUVve7D"
    openai.api_key = config.OPENAI_API_KEY
    openai.Model.list()

    gptResponse = openai.ChatCompletion.create(
    model = "gpt-3.5-turbo", messages = [{"role": "user", "content": "ITEM #470974 사료관리법에 의한표시사항 제품명: 커클랜드 시그니춰 네이쳐스 도메인 새언앤 스위트포테이토도그 Kirkland Signature Nature's Domain Salmon Meal & Sweet Potato Formula Dog Food 사료의명칭: 애완육성개시료 6호 성분등록번호: 제 77WLI0001호 사료의용도: 생후 1개월이후 개사료용 사료의 형태: 익스트루전 사료의 종류: 그 밖의 동물·어류용 배합사료 등록성분량: 부지 조단백 조지방 조성유 조회분 칼슘 인 수분 23% 11% 3.7% 9% 0.9% 0.8% 10% 이상 이상 이하 이하 이상 이상 이하 사용한 원료의 명칭: 연어분말,완두콩,고구마,감자,카놀리유.어분.감자 식이섬유,완두콩백질,천연착향료,아마씨,정제소금.염화콜린.이눌린 토마토,블루베리,라즈베리,유카추출물,단백질화합물철,아연,망간, 구리),황산철/아인/만간/구리,산화망간,요오드칼륨,0셀렌산나트륨, 비타민E.비타민A.비타민D,니코틴산,영산,비타민E6.비타민B2, 비타민B1,비타민E12.판토텐산.비오틴틴비타민C.건조엔테로코커스 훼시엄발효제품,토바실러스 아시도필루스,락토바실러스 카세이. 락토바실러스 플런타령,건조트리코데르마 발효추출물 중량: 15.87kg 원산지 및 수출국: 미국 제조일자: 제품에 별도 표시(년월일순) 페트 유통기한: 제품에 별도 표시일까지(년월일순) 제조원: DIAMOND PETFOODS 수입원: (주)코스트코코리아 전화: 1899-9900 주소:경기도광명시 일직로 40 반품 및 교환장소: 구입처 및 수입원 주의사항 - 직사광선을 피하고 통풍이되는 선선한 장소에 보관 하십시오. (하절기 및 고온다습시 보곤주의) 급여중인 사료는 이물질 혼입방지를 위하여 밀폐하여 주십시오. - 개봉 후 가급적 빨리 급여하여 주시기 바랍니다. - 반추가축에게 먹이지 마시고, 개 사료용으로만 급여하시기 바랍니다. 본 제품은 공정거래위원회고시 소비자분쟁 해결기준에 의거 교환또는 보상받을 수 있습니다. 라는 텍스트 중에서, 식품의 원재료만 추출해서 줘라. 구분은 쉼표로 해줘."}]
    )
    print(gptResponse)
    return gptResponse


# FastAPI 앱 실행
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
