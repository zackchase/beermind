# deepx
A sequence to sequence to ... recurrent neural network for conversations spanning multiple exchanges

# Getting started
First, make sure you have `virtualenv` installed. The core dependencies are `numpy`, `scipy`, and `theano`.
Please make sure you have the appropriate dependencies to install those.

To set up the `virtualenv`, run the following commands:
```bash
$ virtualenv --no-site-packages venv
$ source venv/activate
$ pip install setuptools pip --upgrade # need newest setuptools/pip
$ pip install -r requirements.txt
```

Then, to install `deepx`, run:
```bash
$ python setup.py develop
```

# Running tests
Yes, we have tests for some reason.

```bash
$ nose tests/
```
