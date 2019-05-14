from setuptools import setup, Extension

setup(name='seqmml',
      version='0.1',
      description='Hyperparameter Optimization based on Sequential Meta Machine Learning.',
      author='Zebin Yang',
      author_email='yangzb2010@hku.hk',
      license='GPL',
      packages=['seqmml','seqmml.pybayopt','seqmml.pysequd', 'seqmml.pybatdoe'],
      install_requires=['joblib', 'numpy', 'pandas', 'scikit-learn', 'tqdm', 'hyperopt','smac', 'pyDOE', 'sobol_seq',
                   'pyunidoe @ git+https://github.com/ZebinYang/pyunidoe.git',
                   'spearmint @ git+https://github.com/ZebinYang/spearmint-lite.git'],
      zip_safe=False)