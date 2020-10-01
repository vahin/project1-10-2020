import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

print("LOADING")

X = np.load("image.npz")["arr_0"]
y = pd.read_csv("labels.csv")

print("SPLITTING")

xTrain, _, yTrain, _ = train_test_split(X, y, random_state=9, train_size=0.55)

newXTrain = xTrain/255

clf = LogisticRegression(solver="saga", multi_class="multinomial")

print("TRAINING")

clf.fit(newXTrain, yTrain.values.ravel())

print("DONE TRAINING")

def predictAlpha(x):
    img = Image.open(x).convert('L').resize((20,33), Image.ANTIALIAS)
    imgScaled = np.clip(img-np.percentile(img, 20), 0, 255)
    imgScaled = np.asarray(imgScaled)/np.max(img)
    return clf.predict(np.array(imgScaled).reshape(1,660))[0]

print("DONE")
