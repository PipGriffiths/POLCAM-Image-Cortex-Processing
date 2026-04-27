from setuptools import setup, find_packages

setup(
    name="pippolcam",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "scikit-image",
        "matplotlib",
        "napari",
        "qtpy"
    ],
    author="Your Name",
    description="Polarisation Camera Image Analysis Suite",
    url="https://github.com/yourusername/PipPolcam",
)
