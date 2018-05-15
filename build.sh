
#!/bin/bash

$PYTHON setup.py install

# Add more build steps here, if they are necessary.
conda install numpy
conda install pandas
conda install scipy
pip install stringcase
pip install git+https://github.com/madrury/basis-expansions.git
pip install git+https://github.com/madrury/regression-tools.git
pip install tqwm

# See
# http://docs.continuum.io/conda/build.html
# for a list of environment variables that are set during the build process.
