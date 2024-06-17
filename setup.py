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
      version='2.2.67',
      description='ML Toolbox',
      author='Hannes Hansen',
      author_email='',
      packages=setuptools.find_packages(),
      python_requires='>=3.5.3',
      install_requires=get_install_requires(),
      extras_require={
          "anomaly_detection": ["scikit-learn==1.3.2", "torch==2.3.1", "mlflow==2.11.1"],
          "peak_shaving": ["scikit-learn==1.3.2", "mlflow==2.11.1", "kneed==0.8.5"],
          "load_shifting": ["scipy==1.13.1", "ray", "mlflow==2.11.1"],
          "estimation": ["darts==0.24.0", "torch==2.0.1", "gluonts==0.13.2"],
          "data": ["ksql-query-builder @ git+https://github.com/SENERGY-Platform/ksql-query-builder", "boto3==1.34.77", "httpx[http2]==0.27.0"]
      }
)