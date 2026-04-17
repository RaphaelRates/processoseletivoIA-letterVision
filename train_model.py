import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class LetterVision:
    def __init__(self):
        self.model = self.build_model()

    def load_data(self):
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

        x_train = x_train.astype("float32") / 255.0
        x_test  = x_test.astype("float32") / 255.0

        x_train = x_train[..., None]
        x_test  = x_test[..., None]

        return (x_train, y_train), (x_test, y_test)

    def build_model(self):
        # model = keras.Sequential([
        #     layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        #     layers.MaxPooling2D((2, 2)),

        #     layers.Conv2D(64, (3, 3), activation='relu'),
        #     layers.MaxPooling2D((2, 2)),

        #     layers.Conv2D(128, (3, 3), activation='relu'),

        #     layers.Flatten(),
        #     layers.Dense(128, activation='relu'),
        #     layers.Dense(10, activation='softmax')
        # ])

        model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def train(self, x_train, y_train):
        self.model.fit(
            x_train, y_train,
            epochs=5,
            batch_size=64,
            validation_split=0.1
        )

    def evaluate(self, x_test, y_test):
        test_loss, test_acc = self.model.evaluate(x_test, y_test, verbose=0)

        y_pred_probs = self.model.predict(x_test, verbose=0)
        y_pred = tf.argmax(y_pred_probs, axis=1)

        cm = tf.math.confusion_matrix(y_test, y_pred)
        print("\n=== MATRIZ DE CONFUSÃO ===")
        print(cm.numpy())

        cm = tf.cast(cm, tf.float32)

        tp = tf.linalg.diag_part(cm)
        fp = tf.reduce_sum(cm, axis=0) - tp
        fn = tf.reduce_sum(cm, axis=1) - tp
        tn = tf.reduce_sum(cm) - (tp + fp + fn)

        precision = tp / (tp + fp + 1e-7)
        recall    = tp / (tp + fn + 1e-7)
        specificity = tn / (tn + fp + 1e-7)
        f1_score  = 2 * precision * recall / (precision + recall + 1e-7)
        accuracy_per_class = (tp + tn) / (tp + tn + fp + fn + 1e-7)

        print("\n=== MÉTRICAS POR CLASSE ===")
        for i in range(10):
            print(
                f"Classe {i}: "
                f"Precision={precision[i]:.4f} | "
                f"Recall={recall[i]:.4f} | "
                f"Specificity={specificity[i]:.4f} | "
                f"F1={f1_score[i]:.4f} | "
                f"Acc={accuracy_per_class[i]:.4f}"
            )

        macro_precision = tf.reduce_mean(precision)
        macro_recall = tf.reduce_mean(recall)
        macro_specificity = tf.reduce_mean(specificity)
        macro_f1 = tf.reduce_mean(f1_score)

        print("\n=== MÉTRICAS GERAIS ===")
        print(f"Loss: {test_loss:.4f}")
        print(f"Acurácia: {test_acc:.4f}")
        print(f"Precision (macro): {macro_precision:.4f}")
        print(f"Recall (macro): {macro_recall:.4f}")
        print(f"Specificity (macro): {macro_specificity:.4f}")
        print(f"F1-score (macro): {macro_f1:.4f}")

def main():
    lv = LetterVision()
    (x_train, y_train), (x_test, y_test) = lv.load_data()
    lv.train(x_train, y_train)
    lv.evaluate(x_test, y_test)

if __name__ == "__main__":
    main()