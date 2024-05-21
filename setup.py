import setuptools

<<<<<<< HEAD
VERSION = '1.0' 
DESCRIPTION = 'DeepMIMO'
LONG_DESCRIPTION = 'DeepMIMOv2 dataset generator library'

# Setting up
setuptools.setup(
        name="DeepMIMO", 
=======
VERSION = '0.2.4' 
DESCRIPTION = 'DeepMIMOv3'
LONG_DESCRIPTION = 'DeepMIMOv3 dataset generator library'

# Setting up
setuptools.setup(
        name="DeepMIMOv3", 
>>>>>>> DeepMIMOv3-python/main
        version=VERSION,
        author="Umut Demirhan, Ahmed Alkhateeb",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        license_files = ('LICENSE.txt', ),
        install_requires=['numpy',
                          'scipy',
<<<<<<< HEAD
                          'tqdm'
=======
                          'tqdm',
                          'matplotlib',
>>>>>>> DeepMIMOv3-python/main
                          ],
        
        keywords=['mmWave', 'MIMO', 'DeepMIMO', 'python', 'Beta'],
        classifiers= [
<<<<<<< HEAD
            "Development Status :: 4 - Beta",
=======
            "Development Status :: 3 - Alpha",
>>>>>>> DeepMIMOv3-python/main
            "Programming Language :: Python :: 3",
            "Operating System :: OS Independent"
        ],
        package_dir={"": "src"},
        packages=setuptools.find_packages(where="src"),
        url='https://deepmimo.net/'
<<<<<<< HEAD
)
=======
)
>>>>>>> DeepMIMOv3-python/main
