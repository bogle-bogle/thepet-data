import numpy as np
import pandas as pd
import pickle
from gensim.models import Word2Vec
from gensim.models import FastText
from sklearn.metrics.pairwise import cosine_similarity
from joblib import Parallel, delayed

# 모델 로드
model_path = "fasttext_clustered_model.model"
loaded_model = FastText.load(model_path)
# 클러스터도 불러오기
with open("ingredient_clusters.pkl", "rb") as file:
    loaded_clusters = pickle.load(file)

# 성분이 앞에 있을 수록 가중치 계산
def calculate_weighted_vector(ingredients, loaded_model, max_weight=1.0, decay_factor=0.95):
    total_weight = 0
    weighted_vector = np.zeros(loaded_model.vector_size)
    for idx, ingredient in enumerate(ingredients):
        if ingredient in loaded_model.wv:
            weight = max_weight * (decay_factor ** idx)
            total_weight += weight
            weighted_vector += weight * loaded_model.wv[ingredient]
    return weighted_vector / total_weight


# 느려서 병렬 처리하려고 함수 따로 만듦
def calculate_similarity(row, user_vector, ingredient_clusters, loaded_model, weight):
    pet_food_ingredients = row['INGREDIENTS'].split(', ')
    pet_food_vector = calculate_weighted_vector(pet_food_ingredients, loaded_model).reshape(1, -1)

    # 유사도 계산
    similarity = cosine_similarity(user_vector, pet_food_vector)[0][0]

    # 사료 제목에서 ingredient_counts의 단어가 포함된 경우 가중치를 부여
    for ingredient in ingredient_clusters:
        if any(clustered_ingredient in row['NAME'] for clustered_ingredient in ingredient_clusters[ingredient]):
            similarity *= weight
            # break  # 하나라도 있으면 break

    return (row['NAME'], similarity)

# 사용자가 입력한 성분 리스트를 바탕으로 유사한 사료를 10개 추천!
# 제목에 가중치 1.2배 줌
def recommend_pet_food(user_ingredients, data, ingredient_clusters, weight=1):
    user_vector = calculate_weighted_vector(user_ingredients, loaded_model).reshape(1, -1)

    # 병렬 처리
    similarities = Parallel(n_jobs=-1)(
        delayed(calculate_similarity)(row, user_vector, ingredient_clusters, loaded_model, weight) for _, row in
        data.iterrows())

    # 유사도가 높은 상위 10개의 사료를 반환
    sorted_recommendations = sorted(similarities, key=lambda x: x[1], reverse=True)[:10]
    return sorted_recommendations


final_cleaned_data_4 = pd.read_csv('final_cleaned_data_4.csv')
user_input_ingredients = ["닭고기", "건조 닭고기", "완두콩", "렌틸콩", "닭 지방", "천연 닭고기 향", "연어 오일", "닭 간", "자연건조 알팔파", "닭 모래주머니",
                          "아마씨", "닭 연골", "치아씨 오일", "페뉴그릭씨", "호박", "코코넛 오일", "호박씨", "크랜베리", "시금치", "비트", "당근", "스쿼시호박",
                          "블루베리", "이눌린", "강황", "타임", "세이지", "로즈마리", "토코페롤 철 단백질 화합물", "구리 단백질 화합물", "망간 단백질 화합물",
                          "아셀렌산 나트륨", "요오드산칼슘", "비타민 E 보충제", "티아민 질산염", "니아신 보충제", "d판토텐산칼슘", "리보플라빈 보충제", "비타민 A 보퉁제",
                          "비타민 D3 보충제", "비타민 B12 보충제", "피리독신 염산염", "엽산", "소금", "건조 페디오코커스 애시디락티시 발효산물", "건조 락토바실러스 아시도필루스 발효산물",
                          "건조 비피도박테리움 롱검 발효산물", "건조 바실러스 코아귤런스 발효산물"]

recommendations = recommend_pet_food(user_input_ingredients, final_cleaned_data_4, loaded_clusters)
for r in recommendations:
    pet_food_info = final_cleaned_data_4[final_cleaned_data_4['NAME'] == r[0]]
    ingredients = pet_food_info['INGREDIENTS'].values[0]
    print(r, ingredients)
