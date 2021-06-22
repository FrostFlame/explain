import os
import random
import string
import uuid
from itertools import islice

from django.core.files import File
from django.shortcuts import render
from dtreeviz.trees import dtreeviz

# Create your views here.
import numpy as np
from joblib import load, dump
from sklearn import preprocessing
from sklearn.inspection import plot_partial_dependence, permutation_importance
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from yellowbrick.features.radviz import RadViz

from explaining.forms import UploadFileForm, PredictForm


def main_page(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            tid = get_random_string(10)
            request.session['tid'] = tid
            os.mkdir('media/input/%s' % tid)

            model = request.FILES['model']
            with open('media/input/%s/model.joblib' % tid, 'wb+') as destination:
                for chunk in model.chunks():
                    destination.write(chunk)
            points = []
            y = []
            with request.FILES['data'].open() as f:
                with open('media/input/%s/data.txt' % tid, 'wb+') as destination:
                    first_line = f.readline()
                    feature_names = first_line.decode('utf-8').strip().split(' ')
                    destination.write(first_line)
                    for line in islice(f, 1, None):
                        params = line.decode().strip().split(' ')
                        points.append(tuple(list(map(lambda x: int(x) if x.isnumeric() else x, params[:-1]))))
                        y.append(params[-1])
                        destination.write(line)
            X = np.array(points)
            class_names = np.unique(y)
            le = preprocessing.LabelEncoder()
            y = le.fit_transform(y)
            graphs, models, important_features, equations = handle_uploaded_file(load(request.FILES['model']), X, y, feature_names, class_names, le)
            predict_form = PredictForm(features=feature_names)

            return render(request, 'explaining/graphs.html', {'graphs': graphs, 'models': models, 'features': important_features, 'equations': equations, 'predict_form': predict_form})
    else:
        form = UploadFileForm()
        return render(request, 'explaining/main_page.html', {'form': form})


def predict(request):
    tid = request.session.get('tid')
    points = []
    y = []
    with open('media/input/%s/data.txt' % tid, 'rb') as f:
        feature_names = f.readline().decode('utf-8').strip().split(' ')
        for line in islice(f, 1, None):
            params = line.decode().strip().split(' ')
            points.append(tuple(list(map(lambda x: int(x) if x.isnumeric() else x, params[:-1]))))
            y.append(params[-1])
    X = np.array(points)
    class_names = np.unique(y)
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)
    with open('media/input/%s/model.joblib' % tid, 'rb') as f:
        model = load(f)
    predict_values = [[int(a[1]) if a[1].isnumeric() else a[1] for a in request.POST.items()][1:]]
    graphs, models, important_features, equations, predictions = handle_uploaded_file(model, X, y, feature_names,
                                                                         class_names, le, predict_values)
    predict_form = PredictForm(features=feature_names)
    return render(request, 'explaining/graphs.html',
                  {'graphs': graphs, 'models': models, 'features': important_features, 'equations': equations,
                   'predict_form': predict_form, 'predictions': predictions})


def handle_uploaded_file(model, X, y, feature_names, class_names, le, predict_values=None):
    graphs = {}
    models = {}
    predictions = {}
    if predict_values:
        predictions['model'] = class_names[model.predict(predict_values)]

    important_features = feature_importance(model, X, y, feature_names)

    most_important = list(important_features.keys())[0:2]
    most_important = dict(zip(most_important, [feature_names.index(a) for a in most_important]))

    graphs['pdps'] = pdp(model, X, feature_names, class_names, most_important, predict_values=predict_values)
    # graphs['pdps'] = []
    graphs['ices'] = pdp(model, X, feature_names, class_names, most_important, True, predict_values)
    # graphs['ices'] = []
    tree_graph, tree_model, tree_prediction = dtree(model, X, feature_names, predict_values)
    # tree_graph, tree_model, tree_prediction = None, None, None
    graphs['dtree'] = tree_graph
    models['dtree'] = tree_model
    if predict_values:
        predictions['dtree'] = class_names[tree_prediction]

    graphs['circle'] = circle(X, y, feature_names, predict_values)

    reg_graph, equations, reg_model, reg_prediction = logistic_regression(model, X, most_important, le, predict_values)

    graphs['regression'] = reg_graph
    models['regression'] = reg_model
    predictions['regression'] = class_names[reg_prediction]

    if predict_values:
        return graphs, models, important_features, equations, predictions
    else:
        return graphs, models, important_features, equations


def circle(X, y, feature_names, predict_values=None):
    plt.figure()
    path = 'media/circle/'
    visualizer = RadViz(classes=['Предмет 1', 'Предмет 2', 'Предмет 3', 'Предмет 4', 'Предмет 5'], features=feature_names)

    visualizer.fit(X, y)
    visualizer.transform(X)
    filename = path + get_random_string() + '.jpg'
    visualizer.show(filename)
    plt.close()
    return filename


def feature_importance(model, X, y, feature_names):
    r = permutation_importance(model, X, y, n_repeats=30, random_state=0)

    features = dict(zip(feature_names, r.importances_mean))
    features = dict(sorted(features.items(), key=lambda item: item[1], reverse=True))
    return features


def pdp(model, X, feature_names, class_names, features, individual=False, predict_values=None):

    graphs = []
    path = 'media/pdp/'

    features = list(features.values())
    for feature in features:
        for cl, cln in zip(model.classes_, class_names):
            fig = plot_partial_dependence(model, X, features=[feature], feature_names=feature_names, kind='individual' if individual else 'average', target=cl)
            if predict_values:
                plt.axvline(x=predict_values[0][feature], c='r')
            filename = path + get_random_string() + '.jpg'
            graphs.append((filename, cln))
            plt.savefig(filename)
            plt.close()

    if not individual:
        for cl, cln in zip(model.classes_, class_names):
            if predict_values:
                plt.scatter(predict_values[0][features[0]], predict_values[0][features[1]], c='black')
            fig = plot_partial_dependence(model, X, features=[(features[0], features[1])], feature_names=feature_names, kind='average', target=cl)
            filename = path + get_random_string() + '.jpg'
            graphs.append((filename, cln))
            plt.savefig(filename)
            plt.close()

    return graphs


def get_random_string(length=10):
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str


def dtree(model, X, feature_names, predict_values=None):
    clf = DecisionTreeClassifier(random_state=0, max_depth=5)
    y = model.predict(X)
    clf.fit(X, y)
    cross_val_score(clf, X, y, cv=10)

    path = 'media/dtree/'
    filename = path + get_random_string() + '.svg'

    viz = dtreeviz(clf, X, y,
                   target_name="target",
                   feature_names=feature_names,
                   class_names=['Предмет 1', 'Предмет 2', 'Предмет 3', 'Предмет 4', 'Предмет 5'])
    viz.save(filename)

    model_name = 'media/models/' + get_random_string() + '.joblib'
    dump(clf, model_name)
    prediction = None
    if predict_values:
        prediction = clf.predict(predict_values)
    return filename, model_name, prediction


def logistic_regression(model, X, features, le, predict_values=None):

    plt.figure()
    y = model.predict(X)
    labels = list(features.keys())
    features = list(features.values())

    q = []
    for elem in X:
        q.append((elem[features[0]], elem[features[1]]))
    X = np.array(q)

    lr = LogisticRegression(solver='saga', max_iter=100, random_state=42,
                            multi_class='ovr', penalty='l1')
    lr.fit(X, y)

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 1  # step size in the mesh
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = lr.predict(np.c_[xx.ravel(), yy.ravel()])


    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1, figsize=(4, 3))
    plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)
    prediction = None
    if predict_values:
        predict_values = [[predict_values[0][features[0]], predict_values[0][features[1]]]]
        prediction = lr.predict(predict_values)
        plt.scatter(predict_values[0][0], predict_values[0][1], c='black')

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.Paired)

    plt.xlabel(labels[0])
    plt.ylabel(labels[1])

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())

    # plt.show()
    path = 'media/regr/'
    filename = path + get_random_string() + '.jpg'
    plt.savefig(filename)

    coefs = lr.coef_
    bias = lr.intercept_
    equations = []
    classes = le.inverse_transform(lr.classes_)
    for b, c, cl in zip(bias, coefs, classes):
        eq = 'Вероятность отношения к классу %s равна 1/(1+exp(-(%f+(%f*%s)+(%f*%s))))' % (cl, b, c[0], labels[0], c[1], labels[1])
        equations.append(eq)

    model_name = 'media/models/' + get_random_string() + '.joblib'
    dump(lr, model_name)

    return filename, equations, model_name, prediction



