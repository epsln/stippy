from setuptools import setup
setup(
    name = 'stippy',
    version = '0.1.0',
    packages = ['stippy'],
    entry_points = {
        'console_scripts': [
            'stippy = stippy.main:main'
        ]
    })
