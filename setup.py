
from setuptools import setup, find_packages

with open('readme.md') as f:
    readme = f.read()

with open('license') as f:
    license = f.read()

setup(
    name='TNML',
    version='0.0.1',
    description='Tensor networks for machine learning course',
    long_description=readme,
    author='Bojan Žunkovič',
    author_email='bojan.zunkovic@fri.uni-lj.si',
    url='https://github.com/qml-tn/tnml-course2021.git',
    license=license,
    packages=["tnml"],
)
