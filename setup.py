from setuptools import setup

with open('README.md', 'r') as fh:
    long_description = fh.read()

setup(
    name='myproject',
    version='0.0.0',
    description='My project',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Me',
    py_modules=['Adversarial_Observation'],
    package_dir={'': 'Adversarial_Observation'},
    install_requires=[
        'torch'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'license: MIT',
        'Operating System :: OS Independent'
    ],
    extras_require={
        'dev': [
            'pytest>=3.7',
        ]
    }
)