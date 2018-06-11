from setuptools import setup, find_packages

setup(
    name='tflibs',
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),
    install_requires=[
        'tensorflow',
        'tqdm',
        'numpy',
        'opencv-python',
        'requests',
    ],
    version='0.2.0',
    description='Libraries for easy bootstrapping TensorFlow project',
    author='Sanghoon Yoon',
    author_email='shygiants@gmail.com',
    url='https://github.com/shygiants/tflibs',
    keywords=['tensorflow', 'libraries'],
    classifiers=[],
)
