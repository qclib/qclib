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
Digits dataset
https://archive.ics.uci.edu/ml/datasets/Pen-Based+Recognition+of+Handwritten+Digits
"""

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

def load(training_size:int, test_size:int, classes=None, features=64, random_seed=42, normalize=True):
    """
    Digits dataset
    https://archive.ics.uci.edu/ml/datasets/Pen-Based+Recognition+of+Handwritten+Digits
    """
    if classes is None:
        classes = list(range(10))

    class_labels = [r'0', r'1', r'2', r'3', r'4', r'5', r'6', r'7', r'8', r'9']
    class_labels = [class_labels[i] for i in classes]

    data = datasets.load_digits()
    # pylint: disable=no-member
    sample_train, sample_test, label_train, label_test = \
        train_test_split(data.data, data.target, test_size=test_size*10,
                                                 random_state=random_seed,
                                                 shuffle=True,
                                                 stratify=data.target)

    # Standardize for gaussian around 0 with unit variance
    std_scale = StandardScaler().fit(sample_train)
    sample_train = std_scale.transform(sample_train)
    sample_test = std_scale.transform(sample_test)

    # Reduce the number of features
    if features < 64:
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
