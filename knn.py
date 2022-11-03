import numpy as np 
from collections import Counter

# Should use the `find_k_nearest_neighbors` function below.
def predict_label(examples, features, k, label_key="is_intrusive"):
    # Write your code here.

    knn = find_k_nearest_neighbors(examples, features, k)
    labels = [examples[elem][label_key] for elem in knn]
    occurence_count = Counter(labels)
    return occurence_count.most_common(1)[0][0]

    
def find_k_nearest_neighbors(examples, features, k):
    # Write your code here.
    z = np.array(features)
  
    res = []
    for key, v in examples.items():
        X = np.array(v['features'])
        dist = X - z
        d = np.linalg.norm(dist, 2)
        res.append([d,key])
                   
    res.sort(key=lambda x:x[0])
    topK = res[:k]

    knn = [pid for dist, pid in topK]
    return knn