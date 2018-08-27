#!/usr/bin/env bash

sudo easy_install pip
pip install --upgrade virtualenv

virtualenv --system-site-packages venv

. ./venv/bin/activate

easy_install -U pip

pip install -e .[dev]