import numpy as np
import pandas as pd
import pickle
from gensim.models import FastText
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor
import time
import config
# 수정
# Load the model and clusters
model_path = config.FASTTEXT_INGREDIENT_MODEL_PATH
loaded_model = FastText.load(model_path)

# 클러스터 로드
cluster_path = config.INGREDIENT_CLUSTER_PATH
with open(cluster_path, "rb") as file:
    loaded_clusters = pickle.load(file)

# Load the data frame
data_path = config.PRODUCT_RAW_DATA_PATH
final_cleaned_data_4 = pd.read_csv(data_path)


def split_ingredients(ingredient_str):
    return ingredient_str.split(", ")


def get_pet_food_vector(row):
    pet_food_ingredients = split_ingredients(row['INGREDIENTS'])
    return calculate_weighted_vector(pet_food_ingredients[:10]).reshape(1, -1)


def calculate_similarity(user_vector, pet_food_vector):
    return cosine_similarity(user_vector, pet_food_vector)[0][0]


def calculate_weighted_vector(ingredients, max_weight=1.0, decay_factor=0.8):
    total_weight = 0
    weighted_vector = np.zeros(loaded_model.vector_size)
    for idx, ingredient in enumerate(ingredients[:10]):
        if ingredient in loaded_model.wv:
            vector = loaded_model.wv[ingredient]

        else:
            closest_ingredient = min(loaded_clusters.keys(
            ), key=lambda k: loaded_model.wv.similarity(ingredient, k))
            vector = loaded_clusters[closest_ingredient]

        weight = max_weight * (decay_factor ** idx)
        total_weight += weight
        weighted_vector += weight * vector

    return weighted_vector / total_weight if total_weight > 0 else weighted_vector


def get_most_similar_top_nine(user_ingredients):
    start = time.time()
    user_vector = calculate_weighted_vector(user_ingredients).reshape(1, -1)

    def calculate_similarity_batch(rows):
        similarities_batch = []
        for _, row in rows.iterrows():
            pet_food_vector = get_pet_food_vector(row)
            similarity = calculate_similarity(user_vector, pet_food_vector)
            similarities_batch.append((row['NAME'], similarity))
        return similarities_batch

    batch_size = 50
    num_batches = len(final_cleaned_data_4) // batch_size + 1
    batches = np.array_split(final_cleaned_data_4, num_batches)

    similarities_batches = []
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(calculate_similarity_batch, batch)
                   for batch in batches]
        for future in futures:
            similarities_batches.extend(future.result())

    sorted_recommendations = sorted(
        similarities_batches, key=lambda x: x[1], reverse=True)[:9]
    result = []
    for r in sorted_recommendations:
        pet_food_info = final_cleaned_data_4[final_cleaned_data_4['NAME'] == r[0]]
        ingredients = pet_food_info['INGREDIENTS'].values[0]
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
