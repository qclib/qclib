# Copyright 2021 qclib project.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Iris dataset
https://archive.ics.uci.edu/ml/datasets/iris
"""

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

def load(training_size:int, test_size:int, classes=None, features=4, random_seed=42, normalize=True):
    """
    Iris dataset
    https://archive.ics.uci.edu/ml/datasets/iris
    """
    if classes is None:
        classes = list(range(3))

    class_labels = [r'Setosa', r'Versicolor', r'Virginica']
    class_labels = [class_labels[i] for i in classes]

    data, target = datasets.load_iris(return_X_y=True)
    sample_train, sample_test, label_train, label_test = \
        train_test_split(data, target, test_size=test_size*3,
                                       shuffle=True,
                                       stratify=target,
                                       random_state=random_seed)

    # Standardize for gaussian around 0 with unit variance
    std_scale = StandardScaler().fit(sample_train)
    sample_train = std_scale.transform(sample_train)
    sample_test = std_scale.transform(sample_test)

    # Reduce the number of features
    if features < 4:
        pca = PCA(n_components=features).fit(sample_train)
        sample_train = pca.transform(sample_train)
        sample_test = pca.transform(sample_test)

    # Scale to the range (0, +1)
    samples = np.append(sample_train, sample_test, axis=0)
    minmax_scale = MinMaxScaler((0, 1)).fit(samples)
    sample_train = minmax_scale.transform(sample_train)
    sample_test = minmax_scale.transform(sample_test)

    # Normalize rows.
    if normalize:
        sample_train = sample_train / \
                       np.linalg.norm(sample_train, axis=1).reshape((len(sample_train),1))
        sample_test = sample_test / \
                      np.linalg.norm(sample_test, axis=1).reshape((len(sample_test),1))

    # Pick training and test size number of samples for each class label
    training_input = {key: (sample_train[label_train == k, :])[:training_size]
                      for k, key in enumerate(class_labels)}
    test_input = {key: (sample_test[label_test == k, :])[:test_size]
                  for k, key in enumerate(class_labels)}

    return sample_train, training_input, test_input, class_labels
