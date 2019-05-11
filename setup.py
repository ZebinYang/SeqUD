from setuptools import setup, Extension

setup(name='SeqMM',
      version='0.1',
      description='Hyperparameter Optimization based on Sequential Meta Modeling.',
      author='Zebin Yang',
      author_email='yangzb2010@hku.hk',
      license='GPL',
      packages=['seqmm','seqmm.pybayopt','seqmm.pysequd', 'seqmm.pybatdoe'],
      install_requires=['joblib', 'numpy', 'pandas', 'scikit-learn', 'tqdm', 'hyperopt','smac', 'pyDOE', 'sobol_seq',
                   'pyUniDOE @ git+https://github.com/ZebinYang/pyUniDOE.git',
                   'spearmint @ git+https://github.com/ZebinYang/spearmint-lite.git'],
      zip_safe=False)