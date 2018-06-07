from setuptools import setup, find_packages

setup(
    name='tflibs',
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),
    version='0.1',
    description='Libraries for easy bootstrapping TensorFlow project',
    author='Sanghoon Yoon',
    author_email='shygiants@gmail.com',
    url='https://github.com/shygiants/tflibs',
    keywords=['tensorflow', 'libraries'],
    classifiers=[],
)
