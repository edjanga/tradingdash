import setuptools
from pathlib import Path

if __name__ == '__main__':

    with open("README.md", "r") as fh:
        long_description = fh.read()

    setuptools.setup(name="tradingDashboard",\
          version="1.0.0",\
          author="Emmanuel Djanga",\
          author_email="emmanuel.djanga@live.be",\
          description="Data visualisation displaying trading strategies performance.",\
          long_description=long_description,\
          long_description_content_type="text/markdown",\
          url="<https://github.com/edjanga/tradingDashboard>",\
          packages=setuptools.find_packages(where='src',\
                                            include=['tradingDashboard*'],\
                                            exclude=['tradingDashboard*'],),\
          classifiers=[

            "Programming Language :: Python :: 3",

            "License :: OSI Approved :: MIT License",

            "Operating System :: OS Independent",

          ],\
          python_requires='>=3.7',\
          package_dir = {'':'src'}

    )