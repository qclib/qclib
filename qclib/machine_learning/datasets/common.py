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
Common dataset preprocessing routine
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

def preprocessing(training_size:int, test_size:int, features:int, max_features:int,
                    data:np.ndarray, class_labels:list, num_classes:int, random_seed=42, normalize=True):
    """
    Common dataset preprocessing routine
    """

    # pylint: disable=no-member
    sample_train, sample_test, label_train, label_test = \
        train_test_split(data.data, data.target, test_size=test_size*num_classes,
                                                 random_state=random_seed,
                                                 shuffle=True,
                                                 stratify=data.target)

    # Standardize for gaussian around 0 with unit variance
    std_scale = StandardScaler().fit(sample_train)
    sample_train = std_scale.transform(sample_train)
    sample_test = std_scale.transform(sample_test)

    # Reduce the number of features
    if features < max_features:
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
    training_input = {key: (sample_train[label_train == key, :])[:training_size]
                      for key in class_labels}
    test_input = {key: (sample_test[label_test == key, :])[:test_size]
                  for key in class_labels}

    return sample_train, training_input, test_input, class_labels
