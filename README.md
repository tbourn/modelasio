# Modelasio
- Extracts data (GR, AC, NEU, RDEP, ACS) from las file and performs curve fitting on them for a given model.
- Trains data and performs feature scaling on GR, AC, NEU, RDEP (independent variables).
- Fits data using Support Vector and Random Forests Regression.
- Returns Coefficient of Determination (r-squared).
- Returns the best values for parameters of the given model.
- Saves plots as png.

| Abbr. |Description|                         
|-------|-----------|
|GR		|Gamma Ray|
|AC		|Sonic Compressional|
|NEU	|Neutron|
|RDEP	|Deep Resistivity|
|ACS	|Sonic Shear|

## Prerequisites
- [Python 3.6.5 or later version](https://www.python.org/)
- [lasio](https://pypi.org/project/lasio/)
- [Pandas](https://pandas.pydata.org/)
- [NumPy](http://www.numpy.org/)
- [Scipy](https://www.scipy.org/)
- [scikit-learn](http://scikit-learn.org/stable/)
