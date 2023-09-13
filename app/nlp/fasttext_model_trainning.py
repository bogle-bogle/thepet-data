# 클러스터링 - 부분 문자열 유사도 이용. 단어 a가 b에 포함되거나 or b가 a에 포함됨
def cluster_ingredients_by_substring(ingredient, ingredients_list):
    similar_ingredients = []
    for other_ingredient in ingredients_list:
        if ingredient in other_ingredient or other_ingredient in ingredient:
            similar_ingredients.append(other_ingredient)
    return similar_ingredients

def create_ingredient_clusters(ingredient_list):
    ingredient_clusters = {}
    for ingredient in ingredient_list:
        ingredient_clusters[ingredient] = cluster_ingredients_by_substring(ingredient, ingredient_list)
    return ingredient_clusters
