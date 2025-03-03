import numpy as np
import scipy.spatial.distance as distance
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

class SimpleShot:
    def __init__(self):
        self.class_prototypes = None
        self.classes = None
    
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.class_prototypes = {
            c: np.mean(X[y == c], axis=0) for c in self.classes
        }
    
    def predict(self, X):
        preds = []
        for x in X:
            distances = {c: np.linalg.norm(x - proto) for c, proto in self.class_prototypes.items()}
            preds.append(min(distances, key=distances.get))
        return np.array(preds)

class Retrieval:
    def __init__(self, database, labels):
        self.database = database
        self.labels = labels
    
    def retrieve(self, query, k=5):
        dists = np.linalg.norm(self.database - query, axis=1)
        indices = np.argsort(dists)[:k]
        return self.labels[indices]
    
    def evaluate(self, queries, query_labels, k_vals=[1, 3, 5]):
        acc_at_k = {k: 0 for k in k_vals}
        mvacc_at_5 = 0
        
        for i, query in enumerate(queries):
            retrieved_labels = self.retrieve(query, k=max(k_vals))
            for k in k_vals:
                if query_labels[i] in retrieved_labels[:k]:
                    acc_at_k[k] += 1
            if np.bincount(retrieved_labels[:5]).argmax() == query_labels[i]:
                mvacc_at_5 += 1
        
        num_queries = len(queries)
        return {k: acc_at_k[k] / num_queries for k in k_vals}, mvacc_at_5 / num_queries

if __name__ == "__main__":
    
    np.random.seed(42)
    num_classes = 5
    num_samples_per_class = 50
    feature_dim = 10

    X_train = np.vstack([np.random.randn(num_samples_per_class, feature_dim) + i for i in range(num_classes)])
    y_train = np.repeat(range(num_classes), num_samples_per_class)

    X_test = np.vstack([np.random.randn(10, feature_dim) + i for i in range(num_classes)])
    y_test = np.repeat(range(num_classes), 10)

    # SimpleShot Evaluation
    simpleshot = SimpleShot()
    simpleshot.fit(X_train, y_train)
    y_pred = simpleshot.predict(X_test)
    print(f"SimpleShot Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    # 20-Nearest Neighbors Evaluation
    knn = KNeighborsClassifier(n_neighbors=20)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    print(f"20-NN Accuracy: {accuracy_score(y_test, y_pred_knn):.4f}")

    # Linear Probing Evaluation
    linear_probe = LogisticRegression(max_iter=1000, C=1.0)
    linear_probe.fit(X_train, y_train)
    y_pred_linear = linear_probe.predict(X_test)
    print(f"Linear Probing Accuracy: {accuracy_score(y_test, y_pred_linear):.4f}")

    # Retrieval Evaluation
    retrieval = Retrieval(X_train, y_train)
    acc_at_k, mvacc_at_5 = retrieval.evaluate(X_test, y_test)
    print(f"Retrieval Acc@K: {acc_at_k}, MVAcc@5: {mvacc_at_5:.4f}")
