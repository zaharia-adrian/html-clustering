import os
import numpy as np
from bs4 import BeautifulSoup
from zss import Node, simple_distance
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score   


def create_node(tag):
    if tag.name:
        node = Node(tag.name)
        for child in tag.children:
            if child.name or child.string.strip():
                node.addkid(create_node(child) if child.name else Node("text"))
        return node

folder_path = "clones/tier3"

html_files = os.listdir(folder_path)

n = len(html_files)
print(n)
edit_distances = np.zeros((n,n))

# encountered problem in folder tier3, added errors="ignore"
# one of the files wasn't .html, so I added extention check
trees = [create_node(BeautifulSoup(open( folder_path + "/" + file, "r", encoding="utf-8").read(),"html.parser")) for file in html_files if file.endswith(".html")] 

for i in range(n):
    for j in range (i+1, n):
        # print(f"Calc dist for {i} and {j}")
        edit_distances[i][j] = edit_distances[j][i] = simple_distance(trees[i],trees[j])

maxK = 2
silhouette_scores = []

for k in range(2, maxK + 1):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(edit_distances)
    silhouette_scores.append(silhouette_score(edit_distances, labels))


bestK = np.argmax(silhouette_scores) + 2

print(bestK)

kmeans = KMeans(n_clusters=bestK, random_state=42, n_init=10)
labels = kmeans.fit_predict(edit_distances)

result = [[] for _ in range(bestK)]

for index, label in enumerate(labels):
    # print(f"{index}, {label}")
    result[label].append(html_files[index])

print(result)
