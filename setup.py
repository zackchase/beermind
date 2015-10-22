from setuptools import setup, find_packages

setup(
    name = "deepx",
    version = "0.0.0",
    author = "Zachary Chase Lipton, Sharad Vikram",
    author_email = "sharad.vikram@gmail.com",
    license = "MIT",
    keywords = "theano",
    packages=find_packages(include=[
        'deepx',
        'dataset',
    ]),
    classifiers=[
],
)
