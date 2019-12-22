# -*- coding: utf-8 -*-
"""
Installs two executables:
    - binarize
    - train_binarizer
"""
import codecs

from setuptools import setup, find_packages

setup(
    name='document-image-binarization',
    version='0.1',
    description='A selectional auto-encoder approach for document image binarization',
    long_description=codecs.open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    author='Antonio-Javier Gallego, Jorge Calvo-Zaragoza',
    author_email='jgallego@dlsi.ua.es, jcalvo@dlsi.ua.es',
    url='https://github.com/ajgallego/document-image-binarization',
    license='GNU General Public License v3.0',
    packages=['binarize'],
    install_requires=open('requirements.txt').read().split('\n'),
    extras_require={
        'demo': ['opencv-python==4.*'],
    },
    include_package_data = True,
    package_data={
        'binarize': ['MODELS/*.h5'],
    },
    entry_points={
        'console_scripts': [
            'binarize = binarize.binarize:main',
            'binarize-demo = binarize.binarize:demo[demo]',
            'train_binarizer = binarize.train:main',
        ],
    },
)
