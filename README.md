# interlo

interlo allows one to quickly create simulations of interstellar object motion through the milky way.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the interlo python package.

```bash
pip install interlo
```

In order to access the simple web interface, please download the content of this git repo and open "index.html".

## Usage

```python
import interlo as iso
from astropy import units as u

stars = iso.Starset(num_stars=5)
stars.get_isos()
stars.integrate()
stars.get_sun()
stars.animate_position()
```