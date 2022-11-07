import numpy as np 
from collections import Counter
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

# Should use the `find_k_nearest_neighbors` function below.
# def predict_label(examples, features, k, label_key="is_intrusive"):
#     # Write your code here.

#     knn = find_k_nearest_neighbors(examples, features, k)
#     labels = [examples[elem][label_key] for elem in knn]
#     occurence_count = Counter(labels)
#     return occurence_count.most_common(1)[0][0]

    
# def find_k_nearest_neighbors(examples, features, k):
#     # Write your code here.
#     z = np.array(features)
  
#     res = []
#     for key, v in examples.items():
#         X = np.array(v['features'])
#         dist = X - z
#         d = np.linalg.norm(dist, 2)
#         res.append([d,key])
                   
#     res.sort(key=lambda x:x[0])
#     topK = res[:k]

#     knn = [pid for dist, pid in topK]
#     return knn


class KNN:
    def __init__(self, k):
        self.k = k 
        self.data = None
        self.labels = None 
    
    def find_k_nearest_neighbors(self, x):
        dists = []
        for idx, pt in enumerate(self.data):
            dist = np.linalg.norm(pt - x, 2)
            dists.append(dist)
        
        k_nn = np.argsort(dists)[:self.k] 
        return k_nn 
    
    def train(self, data, labels):
        self.data = data 
        self.labels = labels

    def predict_label(self, x):
        k_nn = self.find_k_nearest_neighbors(x)
        labels = self.labels[k_nn]
        occurence_count = Counter(labels)
        return occurence_count.most_common(1)[0][0]


def accuracy(preds, targets):
    num_correct = np.sum(preds == targets) 
    return num_correct / len(targets)


def main(k=3):
    wine_data = load_wine()
    data, labels = wine_data.data, wine_data.target
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)

    knn_clf = KNN(k=k)
    knn_clf.train(X_train, y_train)

    preds = [0] * len(X_test)

    for i, datum in enumerate(X_test):
        pred = knn_clf.predict_label(datum)
        preds[i] = pred
    
    accy = accuracy(preds, y_test)
    print(f"Accuracy = {accy}")

if __name__ == "__main__":
    np.random.seed(4)
    k=4
    main(k)