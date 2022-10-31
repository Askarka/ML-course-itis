import networkx as nx
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import pandas as pd
import numpy as np
import random
import math

def get_points_distance(x1, y1, x2, y2):
  return math.sqrt( (x2 - x1)**2 + (y2 - y1)**2 )

class Edge:
  def __init__(self, index1, x1, y1, index2, x2, y2):
    self.index1 = index1
    self.x1 = x1
    self.y1 = y1
    self.index2 = index2
    self.x2 = x2
    self.y2 = y2
    self.distance = get_points_distance(x1, y1, x2, y2)

# Generate clusters
centers = [(0, 4), (5, 5) , (8,2)]
cluster_std = [1.1, 1, 1.2]

X, y = make_blobs(n_samples=30, cluster_std=cluster_std, centers=centers, n_features=2, random_state=42)

plt.scatter(X[y == 0, 0], X[y == 0, 1], s=10, label="Cluster 1")
plt.scatter(X[y == 1, 0], X[y == 1, 1], s=10, label="Cluster 2")
plt.scatter(X[y == 2, 0], X[y == 2, 1], s=10, label="Cluster 3")
plt.title("Scattered data")
plt.show()

df = pd.DataFrame(X, columns = ["X", "Y"] )
print(df.shape)
# how to get point coords
# x, y = df.iloc[index]['X']  ,  df.iloc[index]['Y']

plt.scatter(df.loc[:, "X"], df.loc[:, "Y"], s=10, c='black')


def find_shortest_open_path(df, k):
  df["isolated"] = True
  min_distance = 100
  edges = []
  init_index1 = 0
  init_index2 = 0
  # Finding min distance between 2 points
  for index1 in df.index:
    for index2 in df.index:
      if index1 == index2:
        continue
      dist = get_points_distance(df.iloc[index1]['X'], df.iloc[index1]['Y'], df.iloc[index2]['X'], df.iloc[index2]['Y'])
      if dist < min_distance:
        min_distance = dist
        init_index1 = index1
        init_index2 = index2
  edges.append(
    Edge(
      init_index1, df.iloc[init_index1]['X'], df.iloc[init_index1]['Y'], init_index2, df.iloc[init_index2]['X'],
      df.iloc[init_index2]['Y']
    )
  )
  df.at[init_index1, 'isolated'] = False
  df.at[init_index2, 'isolated'] = False

  while True in df["isolated"].values:
    min_distance = 100
    min_index1 = 0
    min_index2 = 0
    for index1 in df.index:
      if df.at[index1, 'isolated']:
        for index2 in df.index:
          if index1 == index2 or df.at[index2, 'isolated']:
            continue
          dist = get_points_distance(df.iloc[index1]['X'], df.iloc[index1]['Y'], df.iloc[index2]['X'],
                                     df.iloc[index2]['Y'])
          if dist < min_distance:
            min_distance = dist
            min_index1 = index1
            min_index2 = index2
    edges.append(
      Edge(
        min_index1, df.iloc[min_index1]['X'], df.iloc[min_index1]['Y'], min_index2, df.iloc[min_index2]['X'],
        df.iloc[min_index2]['Y']
      )
    )
    df.at[min_index1, 'isolated'] = False
    df.at[min_index2, 'isolated'] = False

  edges = sorted(edges, key=lambda x: x.distance)
  edges = edges[0:len(edges) - k + 1]
  return edges


edges = find_shortest_open_path(df, 3)
print('Found edges')
for edge in edges:
  print(f"({edge.index1}, {edge.index2})")
for edge in edges:
  plt.plot(
    [edge.x1, edge.x2],
    [edge.y1, edge.y2],
    color='blue'
  )
plt.show()


y = [edge.distance for edge in edges]

x = [i for i in range(len(y))]
plt.plot(x, y)
plt.show()