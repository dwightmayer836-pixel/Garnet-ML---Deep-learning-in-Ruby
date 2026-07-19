# Creates tabular csv of N / 60K MNIST images

from sklearn.datasets import fetch_openml
import pandas as pd

mnist = fetch_openml("mnist_784", version=1, as_frame=False)

X = mnist.data.astype("float32") / 255.0
y = mnist.target.astype("int")

df = pd.DataFrame(X[:100])
df.insert(0, "label", y[:100])

df.to_csv("mnist.csv", index=False)
