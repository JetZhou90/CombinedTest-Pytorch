from sklearn.model_selection import train_test_split
from sklearn import neighbors
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import time
import os
import numpy as np
from PIL import Image
import cv2
import argparse
from sklearn.metrics import accuracy_score
from joblib import dump, load

label_list = ["icon", "text"]
n_neighbors = 7
img_size_list = [28, 32, 64]
cls_names = ['Knn', 'MLP', 'Decision Tree', 'Random Forest', 'Adaboost']
clf_list = [neighbors.KNeighborsClassifier(n_neighbors, weights='distance'),
            MLPClassifier(learning_rate='adaptive', max_iter=100),
            DecisionTreeClassifier(
                max_depth=5, criterion="entropy", random_state=0),
            RandomForestClassifier(
                max_depth=5, criterion="entropy", n_estimators=50),
            AdaBoostClassifier(
                learning_rate=.1, n_estimators=50, random_state=0)
            ]


def ensure_folder(folder_path):
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    return folder_path


def pca_train(n_components, x, save_path):
    pca = PCA(n_components)
    reshape_x = x.reshape([-1, n_components])
    pca.fit(reshape_x)
    ensure_folder(save_path)
    pca_save_path = os.path.join(
        save_path, str(int(n_components**0.5))+'_pca.joblib')
    dump(pca, pca_save_path)
    print('PCA Trained Completed')
    return pca


def classifier_train(classifier_name, classifier, x, y, save_path):
    clf = classifier
    clf.fit(x, y)
    ensure_folder(save_path)
    classifier_save_path = os.path.join(
        save_path, '%s.joblib' % (classifier_name))
    dump(clf, classifier_save_path)
    print('%s Trained Completed' % (classifier_name))
    return clf


def model_load(model_path):
    if not os.path.isfile(model_path) or not os.path.exists(model_path):
        raise 'model path is not correct'
    return load(model_path)


def load_data(data_path, img_size, **kwargs):

    def read_img(img_folder_path, label_index):
        images = []
        labels = []
        for img_name in os.listdir(img_folder_path):
            img_path = os.path.join(img_folder_path, img_name)
            img = cv2.imread(img_path, 0)
            img_data = cv2.resize(img, (img_size, img_size))
            # img = Image.open(img_path)
            # img = img.convert('L')
            # resize_img = img.resize([img_size, img_size])
            # img_data = np.array(resize_img)
            images.append(img_data)
            labels.append(label_index)
        return images, labels

    x = []
    y = []
    for i in range(len(label_list)):
        img_folder_path = os.path.join(data_path, label_list[i])
        if not os.path.exists(img_folder_path):
            raise 'There is no %s folder in %s' % (label_list[i], data_path)
        images, labels = read_img(img_folder_path, i)
        x += images
        y += labels

    x = np.array(x)
    y = np.array(y)
    return x, y


def main(args):
    data_path = args.data_path
    saved_model_path = args.saved_model_path
    is_train = bool(args.train)
    use_pca = bool(args.pca)
    if not os.path.exists(data_path):
        raise 'Data folder path is not exist'

    for image_size in img_size_list:
        x, y = load_data(data_path, image_size)
        X_train, X_test, y_train, y_test = train_test_split(
            x, y, test_size=0.3, random_state=42)
        print('load data completed')

        if use_pca:
            n_components = image_size*image_size
            if is_train:
                pca = pca_train(n_components, x, saved_model_path)
                X_train = pca.transform(X_train.reshape([-1, n_components]))
            else:
                pca_save_path = os.path.join(
                    saved_model_path, str(image_size)+'_'+'pca.joblib')
                if not os.path.exists(pca_save_path):
                    raise 'There is no pca model file in %s' % (pca_save_path)
                pca = model_load(pca_save_path)
                print('PCA Loaded Completed')
            X_test = pca.transform(X_test.reshape([-1, n_components]))

        else:
            X_train = X_train.reshape([-1, image_size*image_size])
            X_test = X_test.reshape([-1, image_size*image_size])

        for name, clf in zip(cls_names, clf_list):
            if is_train:
                classifier = classifier_train(
                    str(image_size)+'_'+name, clf, X_train, y_train, saved_model_path)
            else:
                clf_save_path = os.path.join(
                    saved_model_path, str(image_size)+'_' + name+'.joblib')
                if not os.path.exists(clf_save_path):
                    raise 'There is no  model file in %s' % (clf_save_path)
                classifier = model_load(clf_save_path)
                print(name, 'Loaded Completed')

            start = time.time()
            y_pred = classifier.predict(X_test)
            end = time.time()
            cost_time = end - start
            acc = accuracy_score(y_test, y_pred)
            msg = 'model_name: %s, image_size: %d, cost_time: %f, accuracy: %f' % (
                name, image_size, cost_time, acc)
            print(msg)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("-d", "--data_path", default="result",
                      type=str, help="sub-images folder path")
    args.add_argument("-s", "--saved_model_path",
                      default="models/", type=str, help="saved model path")
    args.add_argument("-t", "--train", type=bool, default=False,
                      help='if train the model')
    args.add_argument("-p", "--pca", type=bool, default=True,
                      help='if using pca')
    args = args.parse_args()
    main(args)
