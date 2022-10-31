import numpy as np
import matplotlib.pyplot as plt
import pygame

points = np.empty((0, 2), dtype='f')

c_radius = 2
c_color = (0, 0, 255)
c_thickness = 0
jet_radius = 20
jet_thr = 0.5

bg_color = (255, 255, 255)
(width, height) = (640, 480)
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("#5")

running = True
pushing = False
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            pushing = True
        elif event.type == pygame.MOUSEBUTTONUP:
            pushing = False

    if pushing and np.random.uniform(0, 1) > jet_thr:
        (x, y) = pygame.mouse.get_pos()
        r = np.random.uniform(0, jet_radius)
        phi = np.random.uniform(0, 2 * np.pi)
        coord = [x + r * np.cos(phi), height - y + r * np.sin(phi)]
        points = np.append(points, [coord], axis=0)

    screen.fill(bg_color)
    for point in points:
        pygame.draw.circle(screen, c_color, (int(point[0]), height - int(point[1])), c_radius, c_thickness)
    pygame.display.flip()

pygame.quit()

fig = plt.figure(figsize=(width / 60, height / 60))
plt.scatter(points[:, 0], points[:, 1], c="blue")
plt.show()


class DB_SCAN():

    def __init__(self, dataset, eps=20.0, min_samples=10):
        self.dataset = dataset
        self.eps = eps
        self.min_samples = min_samples
        self.n_clusters = 0
        self.clusters = {0: []}
        self.visited = set()
        self.clustered = set()
        self.labels = np.array([], dtype='i')
        self.fitted = False

    def get_dist(self, list1, list2):
        return np.sqrt(sum((i - j) ** 2 for i, j in zip(list1, list2)))

    def fit(self):
        for P in self.dataset:
            P = list(P)
            if tuple(P) in self.visited:
                continue
            self.visited.add(tuple(P))
            neighbours = self.get_neighbours(P)
            if len(neighbours) < self.min_samples:
                self.clusters[0].append(P)
            else:
                self.expand_cluster(P)
        self.fitted = True

    def get_neighbours(self, P):
        return [list(Q) for Q in self.dataset \
                if self.get_dist(Q, P) < self.eps]

    def expand_cluster(self, P):
        self.n_clusters += 1
        self.clusters[self.n_clusters] = [P]
        self.clustered.add(tuple(P))
        neighbours = self.get_neighbours(P)
        while neighbours:
            Q = neighbours.pop()
            if tuple(Q) not in self.visited:
                self.visited.add(tuple(Q))
                Q_neighbours = self.get_neighbours(Q)
                if len(Q_neighbours) > self.min_samples:
                    neighbours.extend(Q_neighbours)
            if tuple(Q) not in self.clustered:
                self.clustered.add(tuple(Q))
                self.clusters[self.n_clusters].append(Q)
                if Q in self.clusters[0]:
                    self.clusters[0].remove(Q)

    def get_labels(self):
        labels = []
        if not self.fitted:
            self.fit()
        for P in self.dataset:
            for i in range(self.n_clusters + 1):
                if list(P) in self.clusters[i]:
                    labels.append(i)
        self.labels = np.array(labels, dtype='i')
        return self.labels

def get_dist(list1, list2):
    return np.sqrt(sum((i - j) ** 2 for i, j in zip(list1, list2)))


dist = np.array([[get_dist(point, neighbour) for point in points] for neighbour in points])
dist.sort(axis=1)

m = 8
# print('Расстояния до ближайших m соседей: ')
# print(dist[::, 1:m + 1])

avg_dist = np.mean(dist[::, 1:m + 1], axis=1)
avg_dist.sort()
# print('Средние расстояния до ближайших m соседей: ')
# print(avg_dist)

dbscan = DB_SCAN(points, eps=50, min_samples=m)

labels = dbscan.get_labels()

plt.figure()
plt.scatter(points[:,0], points[:,1], c=labels, cmap=plt.cm.Paired)
plt.show()