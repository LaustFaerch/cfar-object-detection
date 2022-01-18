from setuptools import setup, find_packages

setup(
    name='cfar_filters',
    description='Selection of SAR CFAR functions',
    author='lafa',
    author_email='lfa027@uit.no',
    packages=find_packages(),
    python_requires='>=3.8',
    use_scm_version=[],
    setup_requires=[
        'numpy',
        'pandas',
        'numba',
        'scipy'
    ],
    install_requires=[]
)
