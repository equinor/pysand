from setuptools import setup

exec(open('sand/version.py').read())

setup(name='sand',
      version=__version__,
      description='Sand related calculations',
      author='Thorjan Knudsvik',
      author_email='thokn@equinor.com',
      url='https://github.com/equinor/sand',
      packages=['sand'],
      install_requires=['numpy', 'scipy']
)