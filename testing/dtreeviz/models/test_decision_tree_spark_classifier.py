import pytest
from pyspark.ml.classification import DecisionTreeClassificationModel
from dtreeviz.models.spark_decision_tree import ShadowSparkTree
from pyspark.sql import SparkSession
import numpy as np


@pytest.fixture()
def tree_model() -> (DecisionTreeClassificationModel):
    SparkSession.builder \
        .master("local[2]") \
        .appName("dtreeviz_sparkml") \
        .getOrCreate()
    return DecisionTreeClassificationModel.load("fixtures/spark_decision_tree_classifier.model")


@pytest.fixture()
def shadow_dec_tree(tree_model, dataset_spark) -> ShadowSparkTree:
    features = ["Pclass", "Sex_label", "Embarked_label", "Age_mean", "SibSp", "Parch", "Fare"]
    target = "Survived"
    return ShadowSparkTree(tree_model, dataset_spark[features], dataset_spark[target], features, target)


def test_is_fit(shadow_dec_tree):
    assert shadow_dec_tree.is_fit() is True


def test_is_classifier(shadow_dec_tree):
    assert shadow_dec_tree.is_classifier() == True, "Spark decision tree should be classifier"


def test_get_children_left(shadow_dec_tree):
    assert np.array_equal(shadow_dec_tree.get_children_left(),
                          np.array([1, 2, 3, -1, -1, -1, 7, 8, 9, -1, -1, -1, 13, 14, -1, -1, -1]))


def test_get_children_right(shadow_dec_tree):
    assert np.array_equal(shadow_dec_tree.get_children_right(),
                          np.array([6, 5, 4, -1, -1, -1, 12, 11, 10, -1, -1, -1, 16, 15, -1, -1, -1]))


def test_get_node_nsamples(shadow_dec_tree):
    assert shadow_dec_tree.get_node_nsamples(0) == 891, "Node samples for node 0 should be 891"
    assert shadow_dec_tree.get_node_nsamples(1) == 577, "Node samples for node 1 should be 577"
    assert shadow_dec_tree.get_node_nsamples(5) == 559, "Node samples for node 5 should be 559"
    assert shadow_dec_tree.get_node_nsamples(8) == 3, "Node samples for node 3 should be 3"
    assert shadow_dec_tree.get_node_nsamples(12) == 144, "Node samples for node 12 should be 144"
    assert shadow_dec_tree.get_node_nsamples(10) == 2, "Node samples node node 10 should be 2"
    assert shadow_dec_tree.get_node_nsamples(16) == 23, "Node samples for node 16 should be 23"