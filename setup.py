import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="truss",
    version="0.0.25",
    author="Phil Howes",
    author_email="phil@baseten.co",
    description="A seamless bridge from model development to model delivery",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/basetenlabs/truss",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7,<3.11',
)