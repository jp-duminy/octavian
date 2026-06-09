from setuptools import setup, find_packages

setup(
  name='octavian',
  version=0.1,
  url='https://github.com/jszpila314/octavian',
  author='Jakub Szpila',
  author_email='jakub.szpila314@gmail.com',
  packages=find_packages(),
  python_requires='>=3.11',
  install_requires=[
    'numpy', 'pandas', 'scipy', 'astropy', 'unyt', 'h5py'
  ]
)