##packages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing, decomposition
from sklearn.metrics import confusion_matrix
from scipy import stats

##loading/making all the datasets

m1 = [-1, 1]
m2 = [-1, -1]
m3 = [1, 1]
m4 = [1, -1]
cov = [[1, 0], [0, 1]]

g1 = np.random.multivariate_normal(mean = m1, cov = cov, size = 25)
g2 = np.random.multivariate_normal(mean = m2, cov = cov, size = 25)
g3 = np.random.multivariate_normal(mean = m3, cov = cov, size = 25)
g4 = np.random.multivariate_normal(mean = m4, cov = cov, size = 25)

d1 = pd.DataFrame(g1)
d1["label"] = [1]*25
d2 = pd.DataFrame(g2)
d2["label"] = [2]*25
d3 = pd.DataFrame(g3)
d3["label"] = [3]*25
d4 = pd.DataFrame(g4)
d4["label"] = [4]*25

full_data = d1.append(d2.append(d3.append(d4)))
full_data.columns = ["x", "y", "label"]

x = pd.DataFrame(np.random.uniform(-3, 3, 50))
x.columns = ["x"]
y = pd.DataFrame(np.random.uniform(-3, 3, 50))
y.columns = ["y"]
new_data = pd.concat([x, y], axis = 1)
new_data["label"] = ""

wine = pd.read_csv("wine.csv", names = ["category", "alcohol", "malic_acid", "ash", "alcanility", "magnesium", "total_phenols", "flavanoids", "nonflavanoids", "proanthocyanins", "color_intensity", "hue", "diluted_wines", "proline"])
wine = wine[["alcohol", "malic_acid", "ash", "alcanility", "magnesium", "total_phenols", "flavanoids", "nonflavanoids", "proanthocyanins", "color_intensity", "hue", "diluted_wines", "proline", "category"]]

#visualising full_data (which will be used for training)

plt.figure(figsize = (20, 10))
plt.scatter("x", "y", data = full_data, c = "label", cmap = "rainbow", alpha = 2/3, s = 100, edgecolors = "black") # s ... Größe der Punkte
plt.title("Before KNN")
plt.xlabel("x")
plt.ylabel("y")

plt.tight_layout()
plt.savefig("beforeknn.png")
plt.show()

#knn step by step using new_data, k = 5

pred01 = full_data[["x", "y"]]
pred02 = new_data[["x", "y"]]
neighbor = full_data["label"].append([full_data["label"]]*(len(new_data)-1)).reset_index(drop = True)
norm_pred01 = pd.DataFrame(preprocessing.normalize(pred01))
norm_pred02 = pd.DataFrame(preprocessing.normalize(pred02))
train = norm_pred01.append([norm_pred01]*(len(norm_pred02)-1)).reset_index(drop = True)
test = pd.concat([norm_pred02]*len(norm_pred01), axis = 0).sort_index().reset_index(drop = True)
#euclidean distance
step1 = train.sub(test)**2
step2 = step1.sum(axis = 1)
step3 = step2.apply(np.sqrt)
euc_data = pd.concat([step3, neighbor], axis="columns")
euc_data.columns = ["distance", "neighbor"]
new_data["confidence"] = ""
for i in range(50):
    top = list(euc_data[(i*len(pred01)):((i+1)*len(pred01))].sort_values("distance")[:5]["neighbor"])
    if top.count(top[0]) == len(top):
        new_data.iloc[i, 2] = top[0]
        new_data.iloc[i, 3] = 1.00
    else:
        new_data.iloc[i, 2] = stats.mode(top).mode
        new_data.iloc[i, 3] = stats.mode(top).count/5

#visualising the results
        
full_data["data"] = 1
new_data["data"] = 2
v = full_data.append(new_data.drop(["confidence"], axis = 1))
d = v.drop(["data"], axis = 1)

plt.figure(figsize = (20, 10))
plt.scatter("x", "y", data = v, c = "label", cmap = "rainbow", alpha = 2/3, s = 100, linewidths = "data", edgecolors = "black")
plt.title("After KNN")
plt.xlabel("x")
plt.ylabel("y")

plt.tight_layout()
plt.savefig("afterknn.png")
plt.show()

#function for data prep

def prep(data):
    p = len(data.columns)
    target = data.iloc[:, (p-1)].astype("category")
    pred = data.iloc[:, 0:(p-1)]
    if p == 2:
        print("there is only one predictor, pca is not needed.")
    elif p == 3:
        print("there are only two predictors, pca is not needed.")
        x1 = data.iloc[:, 0]
        x2 = data.iloc[:, 1]
        plt.scatter(x1, x2, c = target)
    else:
        #standardising the data
        stand_pred = preprocessing.scale(pred)
        #pca
        pca = decomposition.PCA(n_components = 2)
        pca.fit(stand_pred)
        pca_data = pd.DataFrame(pca.transform(stand_pred), columns = ["PC1", "PC2"])
        #explained variance
        per_var = np.round(pca.explained_variance_ratio_*100, decimals = 1)
        #plotting
        plt.scatter(pca_data.PC1, pca_data.PC2, c = target, alpha = 2/3)
        plt.xlabel("PC1 - {0}%".format(per_var[0]))
        plt.ylabel("PC2 - {0}%".format(per_var[1]))
    return

prep(wine)

#function for knn

def knn(data, k):
    #cleaning
    data.dropna()
    #normalising
    p = len(data.columns)
    pred = data.iloc[:, 0:(p-1)]
    y = data.iloc[:, (p-1)]
    norm_pred = pd.DataFrame(preprocessing.normalize(pred))
    norm_data = pd.concat([norm_pred, y], axis = 1)
    #splitting
    norm_data = norm_data.sample(frac = 1).reset_index(drop = True)
    test_size = 0.25
    train_data = norm_data[:-int(test_size*len(norm_data))]
    test_data = norm_data[:int(test_size*len(norm_data))]
    shuffled_y = train_data.iloc[:, (p-1)]
    #equalising lengths
    new_train = train_data.append([train_data]*(len(test_data)-1)).reset_index(drop = True)
    new_test = pd.concat([test_data]*len(train_data), axis = 0).sort_index().reset_index(drop = True)
    target = pd.DataFrame(shuffled_y.append([shuffled_y]*(len(test_data)-1))).reset_index(drop = True)
    #euclidean distance
    step1 = new_train.subtract(new_test)**2
    step2 = step1.iloc[:, 0:(p-1)].sum(axis = 1)
    step3 = pd.DataFrame(step2.apply(np.sqrt), columns = ["distance"])
    euc_data = pd.concat([step3, target], axis = "columns")
    #finally the actual knn
    test_data["result"] = ""
    test_data["confidence"] = ""
    q = len(test_data.columns)
    n = len(test_data)
    for i in range(n):
        top = euc_data[(i*len(train_data)):((i+1)*len(train))].sort_values("distance")
        top_k = list(top.iloc[:k, -1])
        if top_k.count(top_k[0]) == len(top_k):
            test_data.iloc[i, (q-2)] = top_k[0]
            test_data.iloc[i, (q-1)] = 1.00
        else:
            test_data.iloc[i, (q-2)] = stats.mode(top_k).mode
            test_data.iloc[i, (q-1)] = stats.mode(top_k).count/k
    return test_data

knn3 = knn(wine, 3)
knn5 = knn(wine, 5)

true3 = knn3["category"]
pred3 = knn3["result"]
confusion_matrix(true3, pred3)

true5 = knn5["category"]
pred5 = knn5["result"]
confusion_matrix(true5, pred5)       


