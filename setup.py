from setuptools import setup

exec(open('pysand/version.py').read())

setup(name='pysand',
      version=__version__,
      description='Sand related calculations',
      author='Thorjan Knudsvik',
      author_email='thokn@equinor.com',
      url='https://github.com/equinor/pysand',
      packages=['pysand'],
      install_requires=['numpy', 'scipy']
)