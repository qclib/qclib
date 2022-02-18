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

from sklearn import datasets
from .common import preprocessing # pylint: disable=relative-beyond-top-level

def load(training_size:int, test_size:int, classes=None, features=64, random_seed=42,
                                                                        normalize=True):
    """
    Digits dataset
    https://archive.ics.uci.edu/ml/datasets/Pen-Based+Recognition+of+Handwritten+Digits
    """
    if classes is None:
        classes = list(range(10))

    class_labels = classes

    data = datasets.load_digits()

    return preprocessing(
        training_size, test_size, features, 64, data, class_labels, random_seed, normalize
    )
