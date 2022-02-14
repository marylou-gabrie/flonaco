import setuptools

long_description = """
Python package for sampling experiments on 
mixture of Gaussians, stochastic Allen-Cahn and transition paths.
"""

setuptools.setup(
    name="flonaco",
    version="0.1.1",
    author="Marylou Gabrié, Grant Rotskoff, Eric Vanden-Eijnden",
    author_email="mgabrie@nyu.edu",
    description="python package for sampling with real-nvp flows",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
