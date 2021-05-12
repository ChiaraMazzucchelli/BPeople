import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn import linear_model
from sklearn.metrics import classification_report, confusion_matrix
import itertools
import argparse


def preprocessing_data(path):
    df = pd.read_csv(path)
    dataset = df.loc(df["Skin_R"] == 0)
    data = dataset.drop(columns=["Extension", "Season", "Subgroup"])
    labels = np.array(dataset.pop('Season'))
    return data, labels


def training(x, label, type='random_forest', random_state=None):
    train, test, train_labels, test_labels = train_test_split(x, label,
                                                              stratify=label,
                                                              test_size=0.3,
                                                              random_state=random_state)

    if type == 'random_forest':
        model = RandomForestClassifier(n_estimators=100,
                                       random_state=random_state,
                                       max_features='sqrt',
                                       n_jobs=-1, verbose=1)
    elif type == 'SVM':
        model = svm.SVC(kernel='linear')

    elif type == 'bayes':
        model = GaussianNB()

    elif type == 'logistic_regression':
        model = linear_model.LogisticRegression(multi_class='multinomial', solver='lbfgs')

    model.fit(train, train_labels)

    if type == 'random_forest':
        n_nodes = []
        max_depths = []

        for ind_tree in model.estimators_:
            n_nodes.append(ind_tree.tree_.node_count)
            max_depths.append(ind_tree.tree_.max_depth)

        print(f'Average number of nodes {int(np.mean(n_nodes))}')
        print(f'Average maximum depth {int(np.mean(max_depths))}')

    return model, train, test, test_labels


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Source: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, size=24)
    plt.colorbar(aspect=4)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, size=14)
    plt.yticks(tick_marks, classes, size=14)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.

    # Labeling the plot
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), fontsize=20,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.grid(None)
    plt.tight_layout()
    plt.ylabel('True label', size=18)
    plt.xlabel('Predicted label', size=18)


def evaluate_model(y_test, y_pred, train, model):
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, classes=['Autumn', 'Spring', 'Summer', 'Winter'],
                          title='Season Confusion Matrix')
    features = list(train.columns)
    fi_model = pd.DataFrame({'feature': features,
                             'importance': model.feature_importances_}). \
        sort_values('importance', ascending=False)
    fi_model.head(10)


def create_model(csv_path, type):
    # csv_path = 'nuovo-dataset.csv'
    x, label = preprocessing_data(csv_path)
    # type = 'random_forest'
    model, X_train, X_test, y_test = training(x, label, type, random_state=None)
    y_pred = model.predict(X_test)
    evaluate_model(y_test, y_pred, X_train, model)


parser = argparse.ArgumentParser()

parser.add_argument('--csv_path', help="path del csv")
parser.add_argument('--type',
                    help="tipo di modello tra cui scegliere. scegliere tra: 'random_forest', 'SVM', 'bayes', 'logistic_regression'")
args = parser.parse_args()

create_model(args.csv_path, args.type)
