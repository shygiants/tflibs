#!/usr/bin/env bash

eval "$(pyenv init - )"

pyenv virtualenv --system-site-packages 3.5.2 tflibs
pyenv activate tflibs

pip install -e .[dev]
