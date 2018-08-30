from setuptools import setup, find_packages

setup(
    name='tflibs',
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),
    install_requires=[
        'tqdm',
        'opencv-python',
        'requests',
    ],
    extras_require={
      'dev': [
          'tensorflow==1.10.0'
      ]
    },
    version='0.3.8',
    description='Libraries for easy bootstrapping TensorFlow project',
    author='Sanghoon Yoon',
    author_email='shygiants@gmail.com',
    url='https://github.com/shygiants/tflibs',
    keywords=['tensorflow', 'libraries'],
    classifiers=[],
)
