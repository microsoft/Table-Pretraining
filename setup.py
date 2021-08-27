# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import setuptools

with open("README.md", "r", encoding="utf8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tapex",
    version="0.1",
    author="Qian Liu",
    author_email="qian.liu@buaa.edu.cn",
    description="A small example package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Microsoft/Table-Pretraining",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'transformers>=4.6.0',
        'numpy==1.20.3',
        "fairseq@git+git://github.com/pytorch/fairseq@801a64683164680562c77b688d9ca77fc3e0cea7",
        "records",
        "pandas"
    ],
)
