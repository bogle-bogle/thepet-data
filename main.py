# -*- coding: utf-8 -*-

import pickle
import time

import pandas as pd
from fastapi import FastAPI
from gensim.models import FastText

import config
from app import routes
from nlp.model_trainning_fasttext import create_ingredient_clusters
from nlp.test import get_most_similar_top_ten

app = FastAPI()

app.include_router(routes.router)

# 초기 데이터셋 및 모델 초기화
# STEP 1. pre-trained 모델을 사용하기 어려움 -> 직접 모델 학습 데이터 생성
dataframe = pd.read_csv(config.INGREDIENT_COUNTS)
whole_ingredients_list = dataframe['Ingredient'].tolist()
ingredient_clusters = create_ingredient_clusters(whole_ingredients_list)

# STEP 2. 성분 클러스터들을 FastText 학습시켜 모델로 저장
clustered_ingredients = list(ingredient_clusters.values())
fasttext_clustered_model = FastText(clustered_ingredients, vector_size=100, window=5, min_count=1, workers=4)
fasttext_clustered_model.save(config.FASTTEXT_INGREDIENT_MODEL_PATH)

# STEP 3. 초기화 제대로 되었는지 테스트
with open(config.INGREDIENT_CLUSTER_PATH, "wb") as file:
    pickle.dump(ingredient_clusters, file)

# 테스트
# # 예시 성분 데이터
# user_input_ingredients = "연어분말,완두콩,고구마,감자,카놀리유.어분.감자 식이섬유,완두콩백질,천연착향료,아마씨,정제소금.염화콜린.이눌린 토마토,블루베리,라즈베리,유카추출물,단백질화합물철,아연,망간, 구리),황산철/아인/만간/구리,산화망간,요오드칼륨,0셀렌산나트륨, 비타민E.비타민A.비타민D,니코틴산,영산,비타민E6.비타민B2, 비타민B1,비타민E12.판토텐산.비오틴틴비타민C.건조엔테로코커스 훼시엄발효제품,토바실러스 아시도필루스,락토바실러스 카세이. 락토바실러스 플런타령,건조트리코데르마 발효추출물"
# user_input_ingredients_list = ["닭고기", "건조 닭고기", "완두콩", "렌틸콩", "닭 지방", "천연 닭고기 향", "연어 오일", "닭 간", "자연건조 알팔파", "닭 모래주머니",
#                           "아마씨", "닭 연골", "치아씨 오일", "페뉴그릭씨", "호박", "코코넛 오일", "호박씨", "크랜베리", "시금치", "비트", "당근", "스쿼시호박",
#                           "블루베리", "이눌린", "강황", "타임", "세이지", "로즈마리", "토코페롤 철 단백질 화합물", "구리 단백질 화합물", "망간 단백질 화합물",
#                           "아셀렌산 나트륨", "요오드산칼슘", "비타민 E 보충제", "티아민 질산염", "니아신 보충제", "d판토텐산칼슘", "리보플라빈 보충제", "비타민 A 보퉁제",
#                           "비타민 D3 보충제", "비타민 B12 보충제", "피리독신 염산염", "엽산", "소금", "건조 페디오코커스 애시디락티시 발효산물", "건조 락토바실러스 아시도필루스 발효산물",
#                           "건조 비피도박테리움 롱검 발효산물", "건조 바실러스 코아귤런스 발효산물"]
#
# print("테스트 2 ===================================================")
# start = time.time()
# print(get_most_similar_top_ten(user_input_ingredients_list))
# end = time.time()
# print("소요 시간 : " + str(end - start))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
