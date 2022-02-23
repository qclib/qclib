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
MNIST dataset
http://yann.lecun.com/exdb/mnist/
"""

from dataclasses import dataclass
import numpy as np
from keras.datasets import mnist
from .common import preprocessing # pylint: disable=relative-beyond-top-level

@dataclass
class Dataset:
    data: np.ndarray
    target: np.ndarray

def load(training_size:int, test_size:int, classes=None, features=784, random_seed=42,
                                                                        normalize=True):
    """
    MNIST dataset
    http://yann.lecun.com/exdb/mnist/
    """
    if classes is None:
        classes = list(range(10))

    class_labels = classes

    (training_data, training_labels), (test_data, test_labels) = mnist.load_data()

    data = Dataset(
            np.append(training_data.reshape(60000,784), test_data.reshape(10000,784), axis=0),
            np.append(training_labels, test_labels)
        )

    sample_total, training_input, test_input, class_labels = preprocessing(
            training_size, test_size, features, 784, data, class_labels, 10, random_seed, normalize
        )

    # Completes 2^10 amplitudes.
    for label in training_input:
        training_input[label] = [np.append(d, np.zeros(240)) for d in training_input[label]]

    for label in test_input:
        test_input[label] = [np.append(d, np.zeros(240)) for d in test_input[label]]

    return sample_total, training_input, test_input, class_labels
