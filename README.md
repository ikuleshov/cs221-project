# cs221-project

## Pre-requisites (MacOS)

### Python install

Make sure you are not using the Python version bundled with MacOS (located in /System/Library). Install latest Python distribution (http://docs.python-guide.org/en/latest/starting/install/osx/):

    brew install python

In a new terminal window, try running python and make sure you see the latest version:

    Python 2.7.12 (default, Jun 29 2016, 14:05:02) 
    [GCC 4.2.1 Compatible Apple LLVM 7.3.0 (clang-703.0.31)] on darwin

You may need to run this command if the new Python is not working: 

    brew link --overwrite python

### TensorFlow

Follow the TensorFlow installation guide (https://www.tensorflow.org/versions/r0.11/get_started/os_setup.html#pip-installation):

    sudo easy_install pip
    sudo easy_install --upgrade six
    sudo pip install --upgrade https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-0.11.0rc1-py2-none-any.whl

### External libraries

Install external libraries required by baseline app:

    pip install mahotas
    pip install scikit-image
    pip install scipy

Run the baseline:

    python baseline.py
