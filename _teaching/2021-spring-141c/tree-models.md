---
title: "Tree-based Models"
collection: teaching
permalink: /teaching/2021-spring-141c/tree
---

# STA 141C Big-data and Statistical Computing

## Discussion 8: Tree-based Models

TA: Tesi Xiao

### Decision Tree

- Pros
    + Non-linear classifier
    + Nonparametric
    + Better interpretability with splitting nodes
    + Fast prediction

- Cons
    + Overfitting
    + Slow training

[`class sklearn.tree.DecisionTreeClassifier(*, criterion='gini', splitter='best', max_depth=None)`](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier)

- Criterion: [Gini impurity](https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity) vs. [Information gain](https://en.wikipedia.org/wiki/Decision_tree_learning#Information_gain)

- [Algorithms](https://scikit-learn.org/stable/modules/tree.html#tree-algorithms-id3-c4-5-c5-0-and-cart)


```python
from sklearn.datasets import load_iris
from sklearn import tree
iris = load_iris()
X, y = iris.data, iris.target
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)
```


```python
tree.plot_tree(clf) # visualize the tree
```

![png](tree/output_4_1.png)


### Ensemble Model

(An aggregate of weak learners)

Multiple diverse models are created by using many different modeling algorithms or using different training data sets to predict **one** outcome. The ensemble model aggregates the prediction of each base model and results in once final prediction for the unseen data.


1. **Bagging** (Bootstrap aggregating)

    In the ensemble, each model is created **independently** and votes with equal weight. Bagging trains each model in the ensemble using a **randomly drawn subset** of the training set

    Example: [Random forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html?highlight=random%20forest#sklearn.ensemble.RandomForestClassifier) (multiple decision trees)
    
    `class sklearn.ensemble.RandomForestClassifier(n_estimators=100, *, criterion='gini', max_depth=None)`

2. **Boosting**
    
    Boosting involves **incrementally** (sequentially) building an ensemble by training each new model instance to emphasize the training instances that previous models mis-classified.
    
    The new learner learns from the previous weak learners.

    Example: [Gradient Boosted Decision Trees](https://scikit-learn.org/stable/modules/ensemble.html#gradient-tree-boosting) (GBDT)

### Classifier comparison

[link](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html#sphx-glr-auto-examples-classification-plot-classifier-comparison-py)
