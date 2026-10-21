from setuptools import setup
setup(
    # Needed to silence warnings (and to be a worthwhile package)
    name='colradpy',
    url='https://github.com/johnson-c/ColRadPy',
    author='Curt Johnson',
    author_email='caj0026@auburn.edu',
    # Needed to actually package something
    packages=['colradpy', 'colradpy.atomic'],
    # the NIST level energies of the top-level atomic folder are installed in colradpy/atomic,
    # where get_nist_levels_txt finds them (nist_energies_dir)
    package_dir={'colradpy.atomic': 'atomic'},
    package_data={'colradpy.atomic': ['elements.txt', 'nist_energies/*']},
    # Needed for dependencies
    install_requires=['numpy<2.5','scipy','matplotlib'],
    # *strongly* suggested for sharing
    version='1.1',
    description='python collisional radiative solver',
    # We will also need a readme eventually (there will be a warning)
    # long_description=open('README.md').read(),
)
