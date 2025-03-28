import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from bs4 import BeautifulSoup

os.environ["OMP_NUM_THREADS"] = "8"



def extract_html_features(html_content):
    soup = BeautifulSoup(html_content, "html.parser")

    tags = [tag.name for tag in soup.find_all(True)]
    unique_tags = set(tags)
    num_tags = len(tags)
    num_unique_tags = len(unique_tags)


    def get_max_depth(tag, depth = 0, max_depth = 1000):
        if not tag.name:
            return 1
        if depth == max_depth:
            return depth
        return max([get_max_depth(child, depth + 1) for child in tag.contents], default = 0) # I gave an error when the list was empty, so, default = 0

    dom_depth = get_max_depth(soup)

    strings_list = [string for string in soup.stripped_strings]
    text = " ".join(strings_list)
    num_words = len(text.split()) # len(strings_list) would not be correct
    text_to_html_ratio = num_words / max(num_tags, 1)


    num_styles = len(soup.find_all("style"))
    # for elem in soup.find_all("style"):
    #     print(elem.string)
    classes = [tag.get("class") for tag in soup.find_all(class_ = True)]
    classes = [i for s in classes for i in s] # flatten the 2d array
    num_unique_classes = len(set(classes))

    num_images = len(soup.find_all("img"))
    num_buttons = len(soup.find_all("button"))

    return np.array([
        num_tags, num_unique_tags, 
        dom_depth, 
        num_words, text_to_html_ratio, 
        num_styles, num_unique_classes,
        num_images, num_buttons
    ])


folder_path = "clones/tier3"

html_files = os.listdir(folder_path)

features = []

# encountered problem in folder tier3, added errors = "ignore"
# in tier3 one of the files isn't .html
for file_name in html_files:
    if file_name.endswith(".html"):
        features.append(extract_html_features(open(folder_path + "/" + file_name, "r", encoding="utf-8").read()))

feature_vectors = np.array(features)

scaler = StandardScaler()
feature_vectors = scaler.fit_transform(feature_vectors)

silhouette_scores = []

max_k = 10

for k in range(2, max_k + 1):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features)

    silhouette_scores.append(silhouette_score(features, labels))

best_k = np.argmax(silhouette_scores) + 2

kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
labels = kmeans.fit_predict(feature_vectors)


result = [[] for _ in range(best_k)]

print(best_k)
for index, label in enumerate(labels):
    print(f"{html_files[index]}, {label}")
    result[label].append(html_files[index])

# print(result)
