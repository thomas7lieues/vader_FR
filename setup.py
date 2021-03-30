import codecs
import os
from setuptools import setup, find_packages

HERE = os.path.abspath(os.path.dirname(__file__))
def read(*parts):
    """
    Build an absolute path from *parts* and and return the contents of the
    resulting file.  Assume UTF-8 encoding.
    """
    with codecs.open(os.path.join(HERE, *parts), "rb", "utf-8") as f:
        return f.read()

setup(
  name = 'vaderSentiment_fr',         # How you named your package folder (MyLib)
  packages = find_packages(),   # Chose the same as "name"
  include_package_data=True,
  version = '1.3.4',      # Start with a small number and increase it with every change you make
  license='MIT',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  long_description = read("README.rst"),
  long_description_content_type = 'text/x-rst',   # Give a short description about your library
  author = 'DEPREZ Olivier, EK Thomas',                   # Type in your name
  author_email = 'olivier@7lieues.io, thomas@7lieues.io',      # Type in your E-Mail
  url = 'https://github.com/thomas7lieues/vader_FR',   # Provide either the link to your github or to your website
  download_url = 'https://github.com/thomas7lieues/vader_FR/archive/v_0.1.tar.gz',    # I explain this later on
  keywords = ['Vader_FR', 'vaderSentiment', 'Vader','Sentiment','French','Analysis'],   # Keywords that define your package best
  install_requires=[            # I get to this in a second
          'unidecode',
          'fuzzywuzzy'
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
  ],
)
