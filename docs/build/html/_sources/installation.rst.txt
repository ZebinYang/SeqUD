Installation
=============

Prerequisite
----------------

The following environments are required for SeqUD package:

- Python 3 (anaconda is preferable)
- SWIG >= 3.0
- C++ compiler

The following commands can be helpful for installing the C++ compiler: 

- **Linux**

.. code-block::

    conda install gxx_linux-64 gcc_linux-64 swig

- **Mac**

.. code-block::

    open /Library/Developer/CommandLineTools/Packages/macOS_SDK_headers_for_macOS_10.14.pkg

- **Windows**

Microsoft Visual Studio 14.0 is required, see `python windows compliers`_ for details.

.. _python windows compliers: https://wiki.python.org/moin/WindowsCompilers#Microsoft_Visual_C.2B-.2B-_14.0_with_Visual_Studio_2015_.28x86.2C_x64.2C_ARM.29


- **Colab**

Run the following codes in Colab notebook: 

.. code-block::

    !apt-get install swig3.0
    !ln -s /usr/bin/swig3.0 /usr/bin/swig
    !pip install git+https://github.com/ZebinYang/SeqUD.git

Github Installation
---------------------

You can install the package by the following console command:

.. code-block::

    pip install git+http://github.com/ZebinYang/SeqUD.git
        
        
Manual Installation
---------------------

If git is not available, you can manually install the package by downloading the source codes and then compiling it by hand:

- Download the source codes from http://github.com/ZebinYang/SeqUD.git.

- unzip and switch to the root folder.

- Run the following shell commands to finish installation.

.. code-block::

    pip install -r requirements.txt
    python setup.py install
   