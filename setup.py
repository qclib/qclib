"""
pypi setup
"""

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="qclib",
    version="0.0.8",
    author="Adenilton Silva",
    author_email="ajsilva@cin.ufpe.br",
    description="A quantum computing library using qiskit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/qclib/qclib",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    install_requires=[
        'scipy>=1.7.1',
        'qiskit>=0.18.3',
        'deprecation',
        'graphviz'
    ]
)
