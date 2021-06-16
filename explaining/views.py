import base64
import io
import random
import string
from itertools import islice

import shap as shap
from django import forms
from django.http import HttpResponseRedirect
from django.shortcuts import render

# Create your views here.
import numpy as np
from joblib import load, dump
from sklearn import tree
from sklearn.inspection import plot_partial_dependence, permutation_importance
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from yellowbrick.datasets import load_occupancy
from yellowbrick.features.radviz import radviz

from explaining.forms import UploadFileForm


def main_page(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            points = []
            y = []
            with request.FILES['data'].open() as f:
                feature_names = f.readline().decode('utf-8').strip().split(' ')
                for line in islice(f, 1, None):
                    param1, param2, param3, param4, param5, course = line.decode().split(' ')
                    points.append((int(param1), int(param2), int(param3), int(param4), int(param5)))
                    y.append(int(course))
            X = np.array(points)
            y = np.array(y)
            graphs, tree_model, important_features = handle_uploaded_file(request.FILES['model'], X, y, feature_names)
            return render(request, 'explaining/graphs.html', {'graphs': graphs, 'tree_model': tree_model, 'features': important_features})
    else:
        form = UploadFileForm()
        return render(request, 'explaining/main_page.html', {'form': form})


def handle_uploaded_file(f, X, y, feature_names):
    model = load(f)
    graphs = {}

    important_features = feature_importance(model, X, y, feature_names)

    graphs['pdps'] = pdp(model, X)
    # graphs['pdps'] = []
    # bayes(model, X)
    graphs['ices'] = ice(model, X)
    # graphs['ices'] = []
    tree_graph, tree_model = dtree(model, X)
    # tree_graph, tree_model = None, None
    graphs['dtree'] = tree_graph

    graphs['circle'] = circle(X, y, feature_names)

    # logistic_regression(model, X)

    # graphs.append(log_graph)


    return graphs, tree_model, important_features


def circle(X, y, feature_names):
    path = 'media/circle/'
    fig = radviz(X, y, classes=['Предмет 1', 'Предмет 2', 'Предмет 3', 'Предмет 4', 'Предмет 5'], features=feature_names)
    filename = path + get_random_string() + '.jpg'
    plt.savefig(filename)
    return filename


def feature_importance(model, X, y, feature_names):
    r = permutation_importance(model, X, y, n_repeats=30, random_state=0)

    features = dict(zip(feature_names, r.importances_mean))
    features = dict(sorted(features.items(), key=lambda item: item[1], reverse=True))
    # for i in r.importances_mean.argsort()[::-1]:
    #     if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
    #         features[feature_names[i]] = [r.importances_mean[i], r.importances_std[i]]
    return features


def pdp(model, X):

    features = [1]
    graphs = []
    path = 'media/pdp/'
    fig1 = plot_partial_dependence(model, X, features, kind='average', target=1)
    filename = path + get_random_string() + '.jpg'
    graphs.append(filename)
    plt.savefig(filename)
    # plt.show()
    fig2 = plot_partial_dependence(model, X, features, kind='average', target=2)
    filename = path + get_random_string() + '.jpg'
    graphs.append(filename)
    plt.savefig(filename)
    # plt.show()
    fig3 = plot_partial_dependence(model, X, features, kind='average', target=3)
    filename = path + get_random_string() + '.jpg'
    graphs.append(filename)
    plt.savefig(filename)
    # plt.show()
    fig4 = plot_partial_dependence(model, X, features, kind='average', target=4)
    filename = path + get_random_string() + '.jpg'
    graphs.append(filename)
    plt.savefig(filename)
    # plt.show()
    fig5 = plot_partial_dependence(model, X, features, kind='average', target=5)
    filename = path + get_random_string() + '.jpg'
    graphs.append(filename)
    plt.savefig(filename)
    # plt.show()
    return graphs


def ice(model, X):

    features = [1]
    graphs = []
    path = 'media/pdp/'
    fig1 = plot_partial_dependence(model, X, features, kind='individual', target=1)
    filename = path + get_random_string() + '.jpg'
    graphs.append(filename)
    plt.savefig(filename)
    # plt.show()
    fig2 = plot_partial_dependence(model, X, features, kind='individual', target=2)
    filename = path + get_random_string() + '.jpg'
    graphs.append(filename)
    plt.savefig(filename)
    # plt.show()
    fig3 = plot_partial_dependence(model, X, features, kind='individual', target=3)
    filename = path + get_random_string() + '.jpg'
    graphs.append(filename)
    plt.savefig(filename)
    # plt.show()
    fig4 = plot_partial_dependence(model, X, features, kind='individual', target=4)
    filename = path + get_random_string() + '.jpg'
    graphs.append(filename)
    plt.savefig(filename)
    # plt.show()
    fig5 = plot_partial_dependence(model, X, features, kind='individual', target=5)
    filename = path + get_random_string() + '.jpg'
    graphs.append(filename)
    plt.savefig(filename)
    # plt.show()
    return graphs


def get_random_string(length=10):
    # choose from all lowercase letter
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str


def bayes(model, X):

    y = model.predict(X)

    gnb = GaussianNB()
    gnb.fit(X, y)


def dtree(model, X):
    clf = DecisionTreeClassifier(random_state=0, max_depth=3)
    y = model.predict(X)
    clf.fit(X, y)
    cross_val_score(clf, X, y, cv=10)

    px = 1 / plt.rcParams['figure.dpi']
    plt.subplots(figsize=(1920 * px, 1080 * px))
    graph = tree.plot_tree(clf)
    # plt.show()
    path = 'media/dtree/'
    filename = path + get_random_string() + '.jpg'
    plt.savefig(filename)
    model_name = 'media/models/' + get_random_string() + '.joblib'
    dump(clf, model_name)
    return filename, model_name


def logistic_regression(model, X):
    y = model.predict(X)

    q = []
    for elem in X:
        q.append((elem[0], elem[1]))
    X =np.array(q)

    clf = LogisticRegression(solver='sag', max_iter=100, random_state=42,
                             multi_class='multinomial').fit(X, y)

    # print the training scores
    print("training score : %.3f (%s)" % (clf.score(X, y), 'multinomial'))

    # create a mesh to plot in
    h = .02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    print(clf.intercept_)
    print(clf.coef_)
    # # Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    # # # Put the result into a color plot
    # # Z = Z.reshape(xx.shape)
    # # plt.figure()
    # # plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
    # # plt.title("Decision surface of LogisticRegression (%s)" % 'multinomial')
    # # plt.axis('tight')
    # #
    # # # Plot also the training points
    # # colors = "bry"
    # # for i, color in zip(clf.classes_, colors):
    # #     idx = np.where(y == i)
    # #     plt.scatter(X[idx, 0], X[idx, 1], c=color, cmap=plt.cm.Paired,
    # #                 edgecolor='black', s=20)
    # #
    # # # Plot the three one-against-all classifiers
    # # xmin, xmax = plt.xlim()
    # # ymin, ymax = plt.ylim()
    # # coef = clf.coef_
    # # intercept = clf.intercept_
    #
    # def plot_hyperplane(c, color):
    #     def line(x0):
    #         return (-(x0 * coef[c, 0]) - intercept[c]) / coef[c, 1]
    #
    #     plt.plot([xmin, xmax], [line(xmin), line(xmax)],
    #              ls="--", color=color)
    #
    # for i, color in zip(clf.classes_, colors):
    #     plot_hyperplane(i, color)
    # plt.show()


