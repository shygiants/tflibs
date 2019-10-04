from setuptools import setup, find_packages

setup(
    name='tflibs',
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),
    install_requires=[
        'wheel',
        'tqdm',
        'opencv-python',
        'requests',
        'pandas',
        'pyyaml',
    ],
    extras_require={
      'dev': [
          'tensorflow==2.0.0'
      ]
    },
    version='1.0.0.dev1',
    description='Libraries for easy bootstrapping TensorFlow project',
    author='Sanghoon Yoon',
    author_email='shygiants@gmail.com',
    url='https://github.com/shygiants/tflibs',
    keywords=['tensorflow', 'libraries'],
    classifiers=[
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    python_requires='>=3.5'
)
