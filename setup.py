#!/usr/bin/env python

import setuptools
from pathlib import Path

def get_install_requires():
    """Returns requirements.txt parsed to a list"""
    fname = Path(__file__).parent / 'requirements.txt'
    targets = []
    if fname.exists():
        with open(fname, 'r') as f:
            targets = f.read().splitlines()
    return targets

setuptools.setup(
      name='toolbox',
      version='2.2.90',
      description='ML Toolbox',
      author='Hannes Hansen',
      author_email='',
      packages=setuptools.find_packages(),
      python_requires='>=3.9.0',
      install_requires=get_install_requires(),
      extras_require={
          "anomaly_detection": ["scikit-learn==1.5.0", "torch==2.3.1", "mlflow==3.2.0"],
          "peak_shaving": ["scipy<2", "mlflow==3.2.0", "KDEpy==1.1.13"],
          "load_shifting": ["scipy==1.13.1", "ray==2.24.0", "mlflow==3.2.0"],
          "estimation": ["darts==0.24.0", "torch==2.0.1", "gluonts==0.13.2"],
          "data": ["ksql-query-builder @ git+https://github.com/SENERGY-Platform/ksql-query-builder", "boto3==1.34.77", "httpx[http2]==0.27.0"]
      }
)