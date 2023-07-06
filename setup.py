from setuptools import setup

setup(
    name='interlo',
    version='1.0.2',
    author='Autumn Pearce',
    description='Quickly create simulations of interstellar object motion through the milky way.',
    url='https://github.com/AutumnPearce/interlo',
    keywords='iso, interstellar, gravitational, simulation',
    python_requires='>=3.11',
    install_requires=['matplotlib','numpy','galpy','astropy'],
    extras_require={
        'test': ['pytest', 'coverage'],
        'display': ['IPython', 'celluloid', 'plotly']
    }
)