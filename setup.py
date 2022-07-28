from setuptools import setup

setup(name='dory',
      version='0.1',
      description='A library to deploy networks on MCUs',
      url='https://github.com/pulp-platform/dory/tree/Alessio-refactoring',
      author='Alessio Burrello',
      author_email='alessio.burrello@unibo.com',
      license='MIT',
      packages=setuptools.find_packages(),
	    python_requires='>=3.5',
	    install_requires=[
	        "onnx",
	        "numpy"
	    ],
      zip_safe=False)