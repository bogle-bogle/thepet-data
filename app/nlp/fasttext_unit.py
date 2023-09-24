import math
import time
import numpy as np
import pandas as pd
import pickle
from gensim.models import FastText
from sklearn.metrics.pairwise import cosine_similarity
from joblib import Parallel, delayed
from config import *

# 미리 모델 로드
fasttext_loaded_model = FastText.load(FASTTEXT_INGREDIENT_MODEL_PATH)
with open(INGREDIENT_CLUSTER_PATH, "rb") as file:
    ingredient_clusters = pickle.load(file)

# 성분이 앞에 있을 수록 가중치 계산
def calculate_weighted_vector(ingredients, max_weight=1.0, decay_factor=0.95):
    total_weight = 0
    weighted_vector = np.zeros(fasttext_loaded_model.vector_size)
    for idx, ingredient in enumerate(ingredients):
        if ingredient in fasttext_loaded_model.wv:
            weight = max_weight * (decay_factor ** idx)
            total_weight += weight
            weighted_vector += weight * fasttext_loaded_model.wv[ingredient]
    return weighted_vector / total_weight

# 병렬 처리 함수
def calculate_similarity_batch(rows, user_vector, weight):
    similarities = []
    for _, row in rows.iterrows():
        pet_food_ingredients = row['INGREDIENTS'].split(', ')
        pet_food_vector = calculate_weighted_vector(
            pet_food_ingredients).reshape(1, -1)

        # 유사도 계산
        similarity = cosine_similarity(user_vector, pet_food_vector)[0][0]

        # 사료 제목에서 ingredient_clusters의 단어가 포함된 경우 가중치를 부여
        for ingredient in ingredient_clusters:
            if any(clustered_ingredient in row['NAME'] for clustered_ingredient in ingredient_clusters[ingredient]):
                similarity *= weight
                # break  # 하나라도 있으면 break

        similarities.append((row['NAME'], similarity))
    return similarities

# 사용자가 입력한 성분 리스트를 바탕으로 유사한 사료를 10개 추천!
# 제목에 가중치 1.2배 줌
def get_most_similar_top_nine(user_ingredients, weight=1):
    start = time.time()
    user_vector = calculate_weighted_vector(user_ingredients).reshape(1, -1)
    product_data = pd.read_csv(PRODUCT_RAW_DATA_PATH)

    # 데이터를 작은 배치로 분할
    batch_size = 100  # 적절한 배치 크기를 선택하세요.
    num_batches = len(product_data) // batch_size + 1
    batches = np.array_split(product_data, num_batches)

    # 병렬 처리
    similarities = Parallel(n_jobs=-1)(
        delayed(calculate_similarity_batch)(batch, user_vector, weight) for batch in batches)

    # 병렬 처리 결과를 평탄화
    similarities = [item for sublist in similarities for item in sublist]

    # 유사도가 높은 상위 10개의 사료를 반환
    result = []
    sorted_recommendations = sorted(
        similarities, key=lambda x: x[1], reverse=True)[:9]
    for r in sorted_recommendations:
        pet_food_info = product_data[product_data['NAME'] == r[0]].iloc[0]
        result.append({
            "id": str(pet_food_info['ID']),
            "name": str(pet_food_info['NAME']),
            "price": int(pet_food_info['PRICE']),
            "mainImgUrl": str(pet_food_info['MAIN_IMG_URL']),
            "ingredients": str(pet_food_info['INGREDIENTS']),
            "matchRate": round(float(r[1]) * 100, 2)
        })
    end = time.time()
    ingredients_str = ", ".join(user_ingredients)
    return {"ingredients": ingredients_str, "suggestions": result, "executionTime": str(round((end - start), 2)) + "sec"}
