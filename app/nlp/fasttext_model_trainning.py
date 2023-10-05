import pandas as pd
import pickle
from gensim.models import FastText

file_path = 'app\dataset\ingredient_counts.csv'  # 경로를 실제 파일 경로로 변경해야 합니다.
df = pd.read_csv(file_path)

ingredient_frequency = df.set_index('Ingredient')['Frequency'].to_dict()


def substring_similarity(ingredient, ingredients_list):
    similar_ingredients = []
    for other_ingredient in ingredients_list:
        if ingredient in other_ingredient or other_ingredient in ingredient:
            similar_ingredients.append(other_ingredient)
    return similar_ingredients


sorted_ingredients_list = df['Ingredient'].tolist()

# 성분 별로 유사한 성분 군집 생성
ingredient_clusters = {}
already_clustered = set()

# 빈도수가 높은 순서로 성분을 순회하면서 군집을 생성
for ingredient in sorted_ingredients_list:
    if ingredient not in already_clustered:
        similar_ingredients = substring_similarity(
            ingredient, sorted_ingredients_list)
        ingredient_clusters[ingredient] = similar_ingredients
        # 유사한 성분을 already_clustered에 추가합니다.
        already_clustered.update(similar_ingredients)

# 군집 출력!!!!
example_clusters = {k: ingredient_clusters[k]
                    for k in list(ingredient_clusters.keys())}

for idx, (cluster_name, ingredients) in enumerate(example_clusters.items()):
    frequency = ingredient_frequency.get(cluster_name, 0)
    print(f"{idx+1}. {cluster_name} (빈도수: {frequency}): {ingredients}")

with open("ingredient_clusters.pkl", "wb") as file:
    pickle.dump(ingredient_clusters, file)

# STEP 2. 성분 클러스터들을 FastText 학습시켜 모델로 저장
clustered_ingredients = list(ingredient_clusters.values())
fasttext_clustered_model = FastText(
    clustered_ingredients, vector_size=300, window=5, min_count=1, workers=4)
fasttext_clustered_model.save("app/model/fasttext_clustered_model.model")
