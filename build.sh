#!/usr/bin/env bash

python setup.py bdist_wheel
python3 setup.py bdist_wheel
twine upload --skip-existing --user shygiants dist/*