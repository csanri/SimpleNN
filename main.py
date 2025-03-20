from sklearn.datasets import fetch_openml
from model import SimpleNN

mnist = fetch_openml("mnist_784", version=1, as_frame=False)
X, y = mnist["data"], mnist["target"]

X = X / 255.0
y = y.astype(int)

split = int(len(X) * 0.8)

X_train, X_eval = X[:split].reshape(-1, 28*28), X[split:].reshape(-1, 28*28)
y_train, y_eval = y[:split], y[split:]

model_simple_nn = SimpleNN(
    input_size=28*28,
    hidden_size=256,
    output_size=10,
    n_rounds=1000,
    lr=0.2
)
model_simple_nn.train(X_train, y_train)
pred = model_simple_nn.predict(X_eval)

accuracy = round((y_eval == pred).sum()/len(y_eval)*100, 1)

print(f"Accuracy: {accuracy}%")
