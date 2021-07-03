from setuptools import setup, find_packages


def requirements():
    with open("requirements.txt", "r") as file:
        data = [line.strip() for line in file.readlines()]

    return data


def readme():
    with open("README.md") as file:
        data = file.read()

    return data


keywords = ("deep learning", "tensorflow")

DESCRIPTION = "API on top of tensorflow 2.0.0 with customized layers and functions. Facilitates the usage of callbacks and custom trainig methods."

setup(
    name="tenning",
    version="1.0.6",
    description=DESCRIPTION,
    long_description=readme(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "License :: MIT License",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    url="",
    author="Guilherme Henrique dos Santos",
    author_email="dos_santos_98@hotmail.com",
    keywords=keywords,
    license="MIT",
    packages=find_packages(exclude=['tests', 'models']),
    install_requires=requirements(),
    python_requires='>=3.6',
)
