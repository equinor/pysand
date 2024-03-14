<img src="https://raw.githubusercontent.com/equinor/pysand/master/resources/logo.png" align="center" title="pySand"/>

[![pypi](https://img.shields.io/pypi/v/pysand)](https://pypi.org/project/pysand/)
![tests](https://github.com/equinor/pysand/actions/workflows/ci.yml/badge.svg)
![python](https://img.shields.io/pypi/pyversions/pysand)
[![SCM Compliance](https://scm-compliance-api.radix.equinor.com/repos/equinor/pysand/badge)](https://scm-compliance-api.radix.equinor.com/repos/equinor/pysand/badge)

PySand is a python package with sand management related calculations for oil and gas industry developed by Equinor.
* Acoustic sand detectors standard calibration and quantification
* Basic (black oil) fluid properties
* DNV RP-O501 erosion rate calculations 
* ER probe sand quantification
* Sand transport models

### Installation instructions
```
pip install pysand
```
###### Upgrade

```
pip install pysand --upgrade
```
###### Removal

```
pip uninstall pysand
```

### Usage
Jupyter Notebooks with example usage can be found in the example directory:
* [Acoustic sand detector](examples/asd.ipynb)
* [Erosion](examples/erosion.ipynb)
* [Fluid properties](examples/fluidproperties.ipynb)
* [Sand transport](examples/sand_transport.ipynb)

Bringing them all together:
* [All pySand modules](examples/all_modules.ipynb)

### Contributing
If you want to contribute to the project and make it better, your help
is very welcome. Follow the following instructions and read the 
[CONTRIBUTING.md](CONTRIBUTING.md) file before getting started.
