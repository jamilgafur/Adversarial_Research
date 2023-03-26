from setuptools import setup

setup(
    name='myproject',
    version='0.0.0',
    description='My project',
    author='Me',
    py_modules=['Adversarial_Observation'],
    package_dir={'': 'Adversarial_Observation'},
    install_requires=[
        'torch'
    ]
)