# -*- coding: utf-8 -*-

import pickle
import time

import pandas as pd
from fastapi import FastAPI
from gensim.models import FastText

from app.nlp.fasttext_model_trainning import create_ingredient_clusters
from config import *
from app import routes
app = FastAPI()

app.include_router(routes.router)

# 초기 데이터셋 및 모델 초기화
# STEP 1. pre-trained 모델을 사용하기 어려움 -> 직접 모델 학습 데이터 생성
dataframe = pd.read_csv(INGREDIENT_COUNTS)
whole_ingredients_list = dataframe['Ingredient'].tolist()
ingredient_clusters = create_ingredient_clusters(whole_ingredients_list)

# STEP 2. 성분 클러스터들을 FastText 학습시켜 모델로 저장
clustered_ingredients = list(ingredient_clusters.values())
fasttext_clustered_model = FastText(clustered_ingredients, vector_size=100, window=5, min_count=1, workers=4)
fasttext_clustered_model.save(FASTTEXT_INGREDIENT_MODEL_PATH)
with open(INGREDIENT_CLUSTER_PATH, "wb") as file:
    pickle.dump(ingredient_clusters, file)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
