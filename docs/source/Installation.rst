Installation
=============

Prerequisite
___________________
The required environment for SeqMM package:

- Python 3
- SWIG >= 3.0
- C++ compiler

For different platform, the following commands may be helpful for installing the C++ compiler: 

- For Linux

.. code-block::

    conda install gxx_linux-64 gcc_linux-64 swig

- For Mac

.. code-block::

    open /Library/Developer/CommandLineTools/Packages/macOS_SDK_headers_for_macOS_10.14.pkg

- For Windows

Microsoft Visual Studio 14.0 is required, see `a link`_ for details.

.. _a link: https://wiki.python.org/moin/WindowsCompilers#Microsoft_Visual_C.2B-.2B-_14.0_with_Visual_Studio_2015_.28x86.2C_x64.2C_ARM.29


Github installation
___________________
Currently, we only support installation from our github repository. You can install the package by the following console command:

.. code-block::

    pip install git+http://github.com/ZebinYang/SeqMM.git
        
        
Manual installation
___________________
Currently, we only support installation from our github repository. You can install the package by the following console command:


- First, download the source codes from http://github.com/ZebinYang/SeqMM.git, unzip and switch to the root folder.

- Then, run the following shell commands to finish installation.

.. code-block::

    pip install -r requirements.txt
    python setup.py install
   