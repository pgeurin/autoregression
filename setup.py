from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'long_description.txt'), encoding='utf-8') as f:
    long_description = '\n' + f.read()

setup(
    name='autoregression',
    version='0.0.4',
    description='Series of Data Science Graphs written by Philip Geurin and Matt Drury',
    long_description=long_description,
    url='https://github.com/pgeurin/autoregression',
    author='Philip Geurin and Matt Drury',
    author_email='philip.geurin@gmail.com',
    license='BSD',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics'
    ],
    keywords=['statistics', 'data', 'science', 'datascience'],
    packages=find_packages(),
    # packages=find_packages(),
    install_requires=['numpy', 'pandas', 'scikit-learn', 'matplotlib', 'scipy',
                        'basis_expansions', 'regression_tools',
                        'stringcase', 'tqdm']
)
