import pathlib
from setuptools import setup


def get_version():
    with open('pysand/version.py') as version_file:
        namespace = {}
        exec(version_file.read(), namespace)
        return namespace['__version__']

VERSION = get_version()

# Read the README.md file
home_dir = pathlib.Path(__file__).parent
README = (home_dir / "README.md").read_text()

if __name__ == "__main__":
      setup(
            name='pysand',
            version=VERSION,
            description='Sand management related calculations',
            long_description=README,
            long_description_content_type="text/markdown",
            author='Equinor ASA',
            author_email='stimo@equinor.com',
            license='GNU GPL',
            url='https://github.com/equinor/pysand',
            classifiers=["Programming Language :: Python :: 3"],
            packages=['pysand'],
            install_requires=['pandas', 'numpy>=1.17', 'scipy']
      )