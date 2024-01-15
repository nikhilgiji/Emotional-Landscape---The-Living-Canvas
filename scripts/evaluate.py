import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
#import scikitplot as skplt

def evaluate_model(model, X_valid, y_valid):
    # Replace this with your existing evaluation code
    # For example:
    scores = model.evaluate(X_valid, y_valid, verbose=1)
    print(f"Validation Loss: {scores[0]}")
    print(f"Validation Accuracy: {scores[1]}")

def plot_performance(history):
    # Replace this with your existing performance plotting code
    # For example:
    sns.set()
    fig = plt.figure(0, (12, 4))

    ax = plt.subplot(1, 2, 1)
    sns.lineplot(history.epoch, history.history['accuracy'], label='train')
    sns.lineplot(history.epoch, history.history['val_accuracy'], label='valid')
    plt.title('Accuracy')
    plt.tight_layout()

    ax = plt.subplot(1, 2, 2)
    sns.lineplot(history.epoch, history.history['loss'], label='train')
    sns.lineplot(history.epoch, history.history['val_loss'], label='valid')
    plt.title('Loss')
    plt.tight_layout()

    plt.savefig('epoch_history_dcnn.png')
    plt.show()

def plot_confusion_matrix(y_valid, yhat_valid):
    # Replace this with your existing confusion matrix plotting code
    # For example:
    skplt.metrics.plot_confusion_matrix(np.argmax(y_valid, axis=1), yhat_valid, figsize=(7, 7))
    plt.savefig("confusion_matrix_dcnn.png")

def visualize_random_images(model, X_valid, y_valid):
    # Replace this with your existing code for visualizing random images
    # For example:
    mapper = {
        0: "happy",
        1: "sad",
        2: "neutral",
    }

    np.random.seed(2)
    random_sad_imgs = np.random.choice(np.where(y_valid[:, 1] == 1)[0], size=9)
    random_neutral_imgs = np.random.choice(np.where(y_valid[:, 2] == 1)[0], size=9)

    fig = plt.figure(1, (18, 4))

    for i, (sadidx, neuidx) in enumerate(zip(random_sad_imgs, random_neutral_imgs)):
        ax = plt.subplot(2, 9, i + 1)
        sample_img = X_valid[sadidx, :, :, 0]
        ax.imshow(sample_img, cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"true:sad, pred:{mapper[model.predict_classes(sample_img.reshape(1, 48, 48, 1))[0]]}")

        ax = plt.subplot(2, 9, i + 10)
        sample_img = X_valid[neuidx, :, :, 0]
        ax.imshow(sample_img, cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"t:neut, p:{mapper[model.predict_classes(sample_img.reshape(1, 48, 48, 1))[0]]}")

        plt.tight_layout()
