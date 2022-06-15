# Quasistatic Fracture using Nonliner-Nonlocal Elastostatics with an Explicit Tangent Stiffness Matrix

[![Codacy Badge](https://app.codacy.com/project/badge/Grade/0e3012d373d24b47bdc85345edc46a22)](https://www.codacy.com/gh/diehlpk/AnalyticStiffnessPython/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=diehlpk/AnalyticStiffnessPython&amp;utm_campaign=Badge_Grade) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5484312.svg)](https://doi.org/10.5281/zenodo.5484312)



## Dependencies

* [numpy](https://numpy.org/)
* [scipy](https://www.scipy.org/)
* [matplotlib](https://matplotlib.org/)
* [shapely](https://pypi.org/project/Shapely/)

## Folders 

* ccm : Scripts for the comparison with classical continuums mechanics 
* hard : Scripts for the hard loading 
* soft : Scripts for the soft laoding 
* tensile : Scripts for the tensile test
* potentials : Different potentials studied in the paper
## Usage

```bash
python3 -m venv deps
source deps/bin/activate
pip install -r requirements.txt
```
## References

1. Diehl, Patrick, and Robert Lipton. "Quasistatic Fracture using Nonlinear‐Nonlocal Elastostatics with Explicit Tangent Stiffness Matrix." International Journal for Numerical Methods in Engineering (2021). [Link](https://onlinelibrary.wiley.com/doi/abs/10.1002/nme.7005}, [Preprint](https://doi.org/10.31224/osf.io/3je6b)
