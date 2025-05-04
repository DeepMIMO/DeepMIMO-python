import setuptools

VERSION = '0.2.9'
DESCRIPTION = 'DeepMIMOv3'
LONG_DESCRIPTION = """

This package contains Python code for DeepMIMOv3 generator library. 

## ⚠️ Archival Notice

All future versions are now maintained in the [DeepMIMO Unified Repository](https://github.com/DeepMIMO/DeepMIMO), 

**This package has been archived**, but can still be installed! 

**DeepMIMOv4** is being developed in this repository and will be made public in May 2025.

You can also install deepmimo v2/v3/v4 via the unified PyPI:

`pip install deepmimo==3`
"""

# Setting up
setuptools.setup(
    name="DeepMIMOv3",
    version=VERSION,
    author="Umut Demirhan, Ahmed Alkhateeb",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    license_files=('LICENSE.md',),
    install_requires=[
        'numpy',
        'scipy',
        'tqdm',
        'matplotlib',
    ],
    keywords=['mmWave', 'MIMO', 'DeepMIMO', 'python', 'Beta'],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent"
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    url='https://deepmimo.net/'
)
