import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyterrier-freetext",
    version="0.0.2",
    author="Andrew Parry",
    author_email='a.parry.1{at}.research.gla.ac.uk',
    description="PyTerrier components for free text manipulation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
)