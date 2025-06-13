import os
import glob

import matplotlib.pyplot as plot
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from PIL import Image, ImageOps


POKEMON_RGB_PATH = './pokemon'
POKEMON_GRAY_PATH = './pokemon_processed'


def plot_gallery(images, titles, file_name, h, w, n_col=4):
    plot.clf()
    n_row = len(images) // n_col + (len(images) % n_col != 0)
    plot.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plot.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        if i >= len(images):
            break
        plot.subplot(n_row, n_col, i + 1)
        plot.imshow(images[i].reshape((h, w)), cmap=plot.cm.gray)
        plot.title(titles[i], size=12)
        plot.xticks(())
        plot.yticks(())
    plot.savefig(file_name)


def get_pokemon(h, w):
    """
    Processing data. Resize pokemon images to size h * w and convert them to
    gray scale. For the convenience, save preprocessed images to the other
    folder.
    """
    os.makedirs(POKEMON_GRAY_PATH, exist_ok=True)
    pokemon_names, pokemon_paths = [], []

    for pokemon_name in sorted(os.listdir(POKEMON_RGB_PATH)):
        folder = os.path.join(POKEMON_RGB_PATH, pokemon_name)
        if not os.path.isdir(folder):
            continue

        # make new dir for the pokemon
        new_folder = os.path.join(POKEMON_GRAY_PATH, pokemon_name)
        os.makedirs(new_folder, exist_ok=True)

        # get path of images with file extension png
        paths_png = glob.glob(os.path.join(folder, '*.png'))
        paths_jpg = glob.glob(os.path.join(folder, '*.jpg'))
        paths = paths_png + paths_jpg
        # iterate over existing pokemon's pictures and process each one
        for i, path in enumerate(sorted(paths)):
            img = Image.open(path)
            # TODO: Checkpoint 1.
            # - Convert `img` to grayscale.
            # - Resize `img` to h x w.
            ####
            img = img.convert(???)
            img = ImageOps.fit(???)
            new_path = os.path.join(
                POKEMON_GRAY_PATH, pokemon_name, '%d.jpg' % i)
            img.save(new_path)

    # read pokemon names and paths
    for pokemon_name in sorted(os.listdir(POKEMON_GRAY_PATH)):
        folder = os.path.join(POKEMON_GRAY_PATH, pokemon_name)
        if not os.path.isdir(folder):
            continue
        paths = [os.path.join(folder, f) for f in sorted(os.listdir(folder))]
        pokemon_names.extend([pokemon_name for _ in range(len(paths))])
        pokemon_paths.extend(paths)

    # convert string label to numerical value
    n_pokemon = len(pokemon_paths)
    target_names = list(np.unique(pokemon_names))
    target = np.searchsorted(target_names, pokemon_names)

    # read data
    pokemons = []
    for i, pokemon_path in enumerate(pokemon_paths):
        img = Image.open(pokemon_path)
        pokemon = np.asarray(img, dtype=np.float32)
        pokemons.append(pokemon)
    pokemons = np.array(pokemons)

    # shuffle pokemon
    indices = np.arange(n_pokemon)
    np.random.RandomState(42).shuffle(indices)
    pokemons, target = pokemons[indices], target[indices]
    pokemons = pokemons.reshape(len(pokemons), -1)

    return pokemons, target, target_names


def show_results(y_test, y_pred, target_names):
    ac = accuracy_score(y_test, y_pred)
    pr = precision_score(y_test, y_pred, average='macro')
    rc = recall_score(y_test, y_pred, average='macro')
    print("\nEvaluate on test set:")
    print("  Accuracy : %.2f" % ac)
    print("  Precision: %.2f" % pr)
    print("  Recall   : %.2f" % rc)

    print("\nConfusion Matrix:")
    print(''.join('%10s' % name for name in ([" "] + target_names)))
    for idx, arr in enumerate(confusion_matrix(y_test, y_pred)):
        print('%10s' % target_names[idx], end='')
        for v in arr:
            print('%10s' % str(v), end='')
        print('')
    print('')


def main():
    np.random.seed(1)
    height, width = 200, 200
    X, y, target_names = get_pokemon(height, width)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    # TODO: Checkpoint 4. Apply PCA to the dataset before training the SVM.
    # - Update `n_components` to set the number of eigenvectors.
    ####
    # print("Compute eigen vectors of train set.")
    # n_components = ???
    # pca = PCA(n_components=n_components, whiten=True).fit(X_train)
    # print("Decompose the dataset by eigen vectors.")
    # X_train = pca.transform(X_train)
    # X_test = pca.transform(X_test)
    # eigenpokemons_titles = [
    #     "eigenpokemon %d" % i
    #     for i in range(pca.components_.shape[0])]
    # plot_gallery(pca.components_, eigenpokemons_titles, "PCA", height, width)

    # TODO: Checkpoint 3. Train a SVM classification model.
    # - Update `param_grid` to allow GridSearchCV to find the best parameters.
    ####
    # print("-" * 24 + " SVC " + "-" * 24)
    # param_grid = {
    #     'kernel': ['rbf', 'linear'],
    #     'C': [???],
    #     'gamma': [???],
    # }
    # clf = GridSearchCV(
    #     SVC(class_weight='balanced'), param_grid, cv=3)
    # clf = clf.fit(X_train, y_train)
    # print('Best params  : %s' % clf.best_params_)
    # print('Best accuracy: %.2f' % clf.best_score_)
    # y_pred = clf.predict(X_test)
    # show_results(y_test, y_pred, target_names)

    # TODO: Checkpoint 2. Train a KNN classification model.
    # - Extend 'n_neighbors' to let GridSearchCV find the best value.
    ####
    print("-" * 24 + " KNN " + "-" * 24)
    param_grid = {
        'n_neighbors': [1],
    }
    clf = GridSearchCV(KNeighborsClassifier(), param_grid, cv=3)
    clf = clf.fit(X_train, y_train)
    print('Best params  : %s' % clf.best_params_)
    print('Best accuracy: %.2f' % clf.best_score_)
    y_pred = clf.predict(X_test)
    show_results(y_test, y_pred, target_names)


if __name__ == "__main__":
    main()
