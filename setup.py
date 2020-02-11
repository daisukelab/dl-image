from setuptools import setup
from dlimage import __version__

with open("README.md", "r") as f:
     readme = f.read()

setup(name='dl-image',
      version=__version__,
      license=license,
      install_requires=['numpy'],
      description='dl-image: Packaging image cliche functions',
      author='daisukelab',
      author_email='contact.daisukelab@gmail.com',
      long_description=readme,
      long_description_content_type="text/markdown",
      url='https://github.com/daisukelab/dl-image',
      packages=['dlimage'],
)