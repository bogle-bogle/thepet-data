import pandas as pd
import pickle
from gensim.models import Word2Vec

def substring_similarity(ingredient, ingredients_list):
    similar_ingredients = []
    for other_ingredient in ingredients_list:
        if ingredient in other_ingredient or other_ingredient in ingredient:
            similar_ingredients.append(other_ingredient)
    return similar_ingredients

df = pd.read_csv('ingredient_counts.csv')
ingredients_list = df['Ingredient'].tolist()

# pre-trained를 여러 이유로... 사용할 수가 없어서 직접 모델 학습 데이터를 생성해볼거임
# 성분 별로 유사한 성분 군집 생성을 위해서 부분 문자열 유사도 이용함 (단어 a가 b에 포함 or 그 반대)
ingredient_clusters = {}
for ingredient in ingredients_list:
    ingredient_clusters[ingredient] = substring_similarity(ingredient, ingredients_list)

# 군집 예시
example_clusters = {k: ingredient_clusters[k] for k in list(ingredient_clusters)[:5]}
print(example_clusters)

# 클러스터도 저장시켜야 함
with open("ingredient_clusters.pkl", "wb") as file:
    pickle.dump(ingredient_clusters, file)


# 성분 군집 => Word2Vec 모델 학습
clustered_ingredients = list(ingredient_clusters.values())
word2vec_clustered_model = Word2Vec(clustered_ingredients, vector_size=100, window=5, min_count=1, workers=4)

# 모델 저장
model_path = "word2vec_clustered_model.model"
word2vec_clustered_model.save(model_path)

# 모델 테스트 해보기
similar_words = word2vec_clustered_model.wv.most_similar("닭고기", topn=10)
for similar_word, similarity in similar_words:
    print(f"{similar_word}: {similarity}")
