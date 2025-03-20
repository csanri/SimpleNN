import numpy as np


class SimpleNN:
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        n_rounds=100,
        lr=0.1
    ):
        self.n_rounds = n_rounds
        self.lr = lr

        # Initialize weights
        self.w1 = np.random.randn(hidden_size, input_size) * 0.01
        self.b1 = np.zeros((hidden_size, 1))
        self.w2 = np.random.randn(output_size, hidden_size) * 0.01
        self.b2 = np.zeros((output_size, 1))

    def forward(self, x):
        x = x.reshape(x.shape[0], -1).T  # (input_size, batch_size)
        self.x = x

        # Layer 1
        self.l1 = self.w1 @ self.x + self.b1  # (hidden_size, batch_size)
        self.y1 = self.ReLU(self.l1)

        # Layer 2
        self.l2 = self.w2 @ self.y1 + self.b2  # (output_size, batch_size)
        self.y2 = self.softmax(self.l2)

        return self.y2

    def backward(self, y_true):
        batch_size = self.x.shape[1]
        y_true_one_hot = self.one_hot(y_true).T  # (output_size, batch_size)

        # Output output gradients
        dy2 = (self.y2 - y_true_one_hot)  # (output_size, batch_size)
        dw2 = (1/batch_size) * dy2 @ self.y1.T  # (output_size, hidden_size)
        db2 = (1/batch_size) * np.sum(dy2, axis=1, keepdims=True)

        # Hidden layer gradients
        dy1 = self.w2.T @ dy2  # (hidden_size, batch_size)
        dRelu = (self.l1 > 0).astype(float)
        dl1 = dy1 * dRelu

        dw1 = (1/batch_size) * dl1 @ self.x.T
        db1 = (1/batch_size) * np.sum(dl1, axis=1, keepdims=True)

        # Update parameters
        self.w2 -= self.lr * dw2
        self.b2 -= self.lr * db2
        self.w1 -= self.lr * dw1
        self.b1 -= self.lr * db1

    def one_hot(self, labels, num_classes=10):
        return np.eye(num_classes)[labels]

    def ReLU(self, x):
        return np.maximum(0, x)

    def softmax(self, x):
        exps = np.exp(x - np.max(x, axis=0))
        return exps / np.sum(exps, axis=0, keepdims=True)

    def train(self, x, y):
        print(f"Training started for {self.n_rounds} rounds...")
        print("-" * 28)
        for i in range(self.n_rounds):
            y_pred = self.forward(x)
            self.backward(y)
            loss = self.loss_fn(y, y_pred)
            print("%-7s %5d | %-5s %.4f" % ("Round:", i, "Loss:", loss))
            print("-" * 28)
        print("Finished training the model finished")

    def predict(self, x):
        y_pred = self.forward(x)
        return np.argmax(y_pred, axis=0)

    # Cross-entropy loss function
    def loss_fn(self, y_true, y_pred):
        correct_class_probs = y_pred[y_true, np.arange(y_true.size)]
        return -np.mean(np.log(correct_class_probs + 1e-10))
