from setuptools import setup

setup(
    name='phoenix_drone_simulation',
    version='1.2.0',
    install_requires=[
        'numpy',
        'gym',
        'pybullet',
        'stable-baselines3',
        'torch',
        'scipy>= 1.4',
        'mpi4py',
        'deap',
    ]
)
