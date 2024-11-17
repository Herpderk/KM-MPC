from setuptools import setup, find_packages

setup(
    name='km_mpc',
    version='0.0.0',
    python_requires='>3.12',
    packages=find_packages(include=[
        'km_mpc',
        'km_mpc.*',
    ]),
    install_requires=[
        'casadi==3.6.7',
        'numpy==2.1.2',
        'scipy==1.14.1',
        'matplotlib==3.9.2',
        'PyQt6==6.7.1',
        'compress_pickle==2.1.0',
    ]
)
