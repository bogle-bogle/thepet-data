import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

data = pd.read_csv('PRODUCTS.csv', low_memory=False)
print(data.head(2))

# overview 열에 존재하는 모든 결측값을 전부 카운트하여 출력
print('overview 열의 결측값의 수:', data['INGREDIENTS'].isnull().sum())

# 결측값을 빈 값으로 대체
data['INGREDIENTS'] = data['INGREDIENTS'].fillna('')

# 행렬의 크기를 출력
tfidf = TfidfVectorizer(stop_words=None)
tfidf_matrix = tfidf.fit_transform(data['INGREDIENTS'])
print('TF-IDF 행렬의 크기(shape) :', tfidf_matrix.shape)

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
print('코사인 유사도 연산 결과 :', cosine_sim.shape)

# 상품ID 입력 => 가장 INGREDIENTS가 유사한 10개의 상품 찾아내는 함수
def get_recommendations(id, cosine_sim=cosine_sim):
    # 해당 상품과, 그에 대한 모든 상품의 유사도를 가져온다.
    sim_scores = list(enumerate(cosine_sim[id]))

    # 유사도에 따라 상품들을 정렬
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # 가장 유사한 10개의 상품들을 받아온다. (선택한 상품 포함)
    sim_scores = sim_scores[:11]

    # 가장 유사한 10개의 상품의 인덱스와 유사도를 얻는다.
    similarProductsIndices = [idx[0] for idx in sim_scores]
    similarityScores = [idx[1] for idx in sim_scores]

    print("==============================")
    results = []
    for index, score in zip(similarProductsIndices, similarityScores):
        product_id = data['ID'].iloc[index]  # 상품의 ID를 가져옴. 'ID'는 상품 ID를 담고 있는 열 이름이라고 가정합니다.
        # 유사도를 백분율로 변환
        percentage_similarity = score * 100
        results.append((product_id, data['INGREDIENTS'].iloc[index], percentage_similarity))

    return results

# 위 함수 실행
# for product_id, ingredient, similarity in get_recommendations(100, cosine_sim=cosine_sim):
#     print(f"Product ID: {product_id}, INGREDIENTS: {ingredient}, Similarity: {similarity:.2f}%")


from sklearn.metrics.pairwise import cosine_similarity

from sklearn.metrics.pairwise import cosine_similarity


def get_recommendations_with_new_data(new_data, tfidf, tfidf_matrix, data):
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
        product_id = data['ID'].iloc[index]
        ingredients = data['INGREDIENTS'].iloc[index]
        results.append((product_id, ingredients, score * 100))  # 유사도를 퍼센트로 변환

    return results

# ID가 32인 행의 'INGREDIENTS' 칼럼 값 선택
new_data = "힐스 프리스크립션 다이어트 c/d 멀티케어 제품의 원재료는 통옥수수, 계육분, 돼지지방, 옥수수글루텐박, 대두밀런, 계란, 대두박, 천연향료, 닭간, 밀글루텐, 대두유젖산아마씨, 천연향료돼지간, L-라 어유, 염화칼륨, 정제소금, 염화콜린, 구연산칼륨, 비타민제(비타민E, 나이아신), 타우린미량광물질류합제, 천연향료, 베타카로틴입니다."

# get_recommendations 함수에 new_data 전달
recommendations = get_recommendations_with_new_data(new_data, tfidf, tfidf_matrix, data)

# 출력
for product_id, ingredient, similarity in recommendations:
    print(f"Product ID: {product_id}, INGREDIENTS: {ingredient}, Similarity: {similarity:.2f}%")