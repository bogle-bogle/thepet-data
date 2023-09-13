import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import PRODUCT_TMP_DATA_PATH

# 1. 파일 불러오기
data = pd.read_csv(PRODUCT_TMP_DATA_PATH, low_memory=False)

# 2. INGREDIENTS가 Null인 결측값을 빈 값으로 대체
data['INGREDIENTS'] = data['INGREDIENTS'].fillna('')

# 3. TF-IDF 행렬 생성
tfidf = TfidfVectorizer(stop_words=None)
tfidf_matrix = tfidf.fit_transform(data['INGREDIENTS'])
print('TF-IDF 행렬의 크기(shape) :', tfidf_matrix.shape)     # 행렬의 크기 출력 : 행, 열 => a개의 행을 표현하기 위해 b개의 단어가 사용됨

# 4. TF-IDF 행렬의 코사인 유사도 계산
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def get_recommendations_with_new_data(new_data):
    # 새로운 데이터를 TF-IDF 벡터로 변환
    new_tfidf = tfidf.transform([new_data])

    # 새로운 데이터와 기존 데이터 간의 코사인 유사도 계산
    cosine_sim_new = cosine_similarity(new_tfidf, tfidf_matrix)

    # 유사도 점수를 기반으로 상품들을 정렬
    sim_scores = list(enumerate(cosine_sim_new[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # 가장 유사한 10개의 상품들을 받아온다.
    sim_scores = sim_scores[:10]

    # 가장 유사한 10개의 상품의 인덱스와 유사도를 얻는다.
    similarProductsIndices = [idx[0] for idx in sim_scores]
    similarityScores = [idx[1] for idx in sim_scores]

    results = []
    for index, score in zip(similarProductsIndices, similarityScores):
        product_id = int(data['ID'].iloc[index])
        product_name = data['NAME'].iloc[index]
        product_price = int(data['PRICE'].iloc[index])
        product_main_img_url = data['MAIN_IMG_URL'].iloc[index]
        ingredients = data['INGREDIENTS'].iloc[index]
        # 결과를 딕셔너리 형태로 저장
        results.append({
            "id": product_id,
            "name": product_name,
            "price": product_price,
            "mainImgUrl": product_main_img_url,
            "ingredients": ingredients,
            "similarity": float(round(score * 100, 2))
        })
    return results