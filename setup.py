from setuptools import setup, find_packages

path_to_readme = "README.md"

if path_to_readme is not None:
    with open(path_to_readme, "r") as desc_file:
        long_description = desc_file.read()
else:
    long_description = ''

path_to_version_file = "./structguy/_version.py"

with open(path_to_version_file) as version_file:
    exec(version_file.read().strip())

setup(
    name="StructGuy",
    version=__version__,
    description="Structur-Guided Variant Effect Prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='LGPL-2.1',
    author="Alexander Gress",
    maintainer="Alexander Gress",

    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    packages=find_packages(),
    setup_requires=['setuptools_scm'],
    include_package_data=True,
    install_requires=[
        "scikit-learn==1.5.0",
        "xgboost==2.0.3",
        "torch==2.8.0",
        "fgclustering==2.0.2",
        "shap==0.48.0",
        "supertree==0.5.5",
        "matplotlib==3.10.6"
    ],

    package_data = {
        "": [
            'resources/psic/psic.c',
            'resources/psic/Makefile',
            'resources/psic/Blosum62.txt',
            'resources/psic/psic_wrapper.py'
            ]
    },
    exclude_package_data={
        '': [
            'resources/search_db_settings.conf'
        ]
    },
    python_requires=">=3.8, <4",
    keywords="bioinformatics",
    entry_points={
        "console_scripts": ["structguy = structguy.structguy_main:main"],
    },
)
