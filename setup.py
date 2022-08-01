from setuptools import setup, Extension
import os


setup(name='dory',
      version='0.1',
      description='A library to deploy networks on MCUs',
      url='https://github.com/pulp-platform/dory/',
      author='Alessio Burrello',
      author_email='alessio.burrello@unibo.com',
      license='MIT',
      packages=setuptools.find_packages(),
	    python_requires='>=3.5',
	    install_requires=[
	        "onnx",
	        "numpy",
              "ortools",
              "mako"
	    ],
      package_data={"": ['Makefile*'], "": ['*.[json,c,h]']},
      include_package_data=True,
      zip_safe=False)