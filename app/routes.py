from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from utils import extract_full_content_with_ocr, extract_foods_with_gpt
from cosine_similarity import get_recommendations_with_new_data

router = APIRouter()

class ImgItem(BaseModel):
    imgUrl: str

class TxtItem(BaseModel):
    content: str

@router.post("/ocr")
async def send_request_to_clova(item: ImgItem):
    return extract_full_content_with_ocr(item)

@router.post("/gpt")
async def send_request_to_gpt(item: TxtItem):
    return extract_foods_with_gpt(item)

@router.post("/extract")
async def extract_foods_from_img(item: ImgItem):
    ocrResult = extract_full_content_with_ocr(item)
    gptResult = extract_foods_with_gpt(ocrResult)
    return gptResult

@router.post("/calculate-similarity")
async def calculate_similarity(item: TxtItem):
    results = get_recommendations_with_new_data(item.content)
    return results

@router.post("/convert-to-similarity")
async def calculate_img_to_similarity(item: ImgItem):
    ocrResult = extract_full_content_with_ocr(item)
    gptResult = extract_foods_with_gpt(ocrResult)
    finalResult = get_recommendations_with_new_data(gptResult)
    return finalResult