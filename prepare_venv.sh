#!/usr/bin/env bash

sudo easy_install pip
pip install --upgrade virtualenv

virtualenv --system-site-packages venv -p python3

. ./venv/bin/activate

easy_install -U pip

pip3 install -e .[dev]