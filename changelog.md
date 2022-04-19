## 1.7.6
* Allowed for all material properties to be retrieved from material function

## 1.7.5
* Added particle density to manifold model and trimmed the validation on same model

## 1.7.4
* Extended the clampon standard step function

## 1.7.3
* Bumped iteration start and maximum number of iterations for Hydro critical velocity calculation

## 1.7.2
* Allow for numpy handling in selected validation functions

## 1.7.1
* Added gitignore and initialisation of test directory

# 1.7
* Added choke operation models from DNVGL RP-O501

## 1.6.5
* sand_rate now returns np.nan instead of None when step is 0 or negative

## 1.6.4
* std_step_clampon and std_step_emerson now returns np.nan instead of None when validate_asd returns True

## 1.6.3
* Added nozzle-valve wall erosion model (derived from CFD study by DNVGL)

## 1.6.2
* Added liquid properties (density and viscosity)

## 1.6.1
* Added sand transport examples notebook

# 1.6
* Added sand transport models

## 1.5.4
* Added exponent to ASD sand rate calculation

## 1.5.3
* Added Travis CI pipeline

## 1.5.2
* Fixed bug in fluid properties test

## 1.5.1
* Updated the example notebooks due to breaking change in v1.5

# 1.5
* Breaking change: Erosion results now in mm/ton instead of mm/year

## 1.4.2
* Choke gallery erosion model test

## 1.4.1
* Added test functions for material properties and angle dependency

# 1.4
* Added material properties to all erosion models

## 1.3.3
* Input validation of ASD calculations
* Input validation of fluid properties
* Test input validation of erosion probes

## 1.3.2
* Erosion models now return NaN when v_m, rho_m, mu_m or Q_s < 0.

## 1.3.1
* Fixes missing input validation for erosion models 

# 1.3
* Added erosion probe sand quantification (incl test and validation)
* Added erosion model for flexible pipes with interlock carcass
* Added erosion model for choke gallery

# 1.2
* Added input validation for erosion models (with tests)

# 1.1
* Fixed calculation error for blinded tees
* Added examples folder and notebooks
* Better readme file with added logo
* Increased decimals on mix viscosity

## 1.0.1
* Modified setup file metadata for PyPi distribution

# 1.0
* Initial public version