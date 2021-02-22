# Analytic stiffness matrix

## Dependencies

* [numpy](https://numpy.org/)
* [scipy](https://www.scipy.org/)
* [matplotlib](https://matplotlib.org/)
* [shapely](https://pypi.org/project/Shapely/)

## Scripts 

* ccm-2D.py : Produces the dispalcement fields obtaiend by classical continuums mechanics 
* bondbased1D.py : Runs the simulation for the one-dimensonal problem
* bondbased2d.py : RUns the simulation for the two-dimensonal linear elastic problem
* bondbased2d-tensile.py " Runs the simulation for the two-dimensional fracture problem using a tensile test geometry
* bondbased2d-plate-soft.py " Runs the simulation for the two-dimensional fracture problem using a pre-cracked square plate with the soft loading (load in force)


## Usage

```bash
python3 -m venv deps
source deps/bin/activate
pip install -r requirements.txt
```
