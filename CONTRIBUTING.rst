CONTRIBUTING
============

Getting started
---------------

1. Fork the GitLab project from https://software.ismll.uni-hildesheim.de/ISMLL-internal/special-interest-group-time-series/tdm.

   Use your personal namespace, e.g. https://software.ismll.uni-hildesheim.de/rscholz/tsdm-dev

2. Clone the forked project locally to your machine. ::

    git clone https://software.ismll.uni-hildesheim.de/rscholz/tsdm-dev
    cd tsdm-dev

3. Checkout the appropriate branch, typically master. ::

    git checkout

4. Setup the virtual environment. You may have to install ``python3.9``. ::

    which python3.9
    sudo apt install python3.9
    python3.9 -m virtualenv venv
    . venv/bin/activate
    pip install -e .

   Or with conda, if you prefer. (You may have to rename ``tables`` and ``torch``). ::

    conda create --name tsdm-dev --file requirements.txt
    conda activate tsdm-dev

  Verify that the installation was successful. ::

    python -c "import tsdm"

5. Create a new working branch. Choose a descriptive name for what you are trying to achieve. ::

    git checkout -b feature-xyz

7. Write your code, bonus points for also adding unit tests.

8. Write descriptive commit messages. Try to keep individual commits easy to understand
   (changing dozens of files, writing 100's of lines of code is not!). ::

    git commit -m '#42: Add useful new feature that does this.'


9. Make sure your changes are parsimonious with the linting and do not break any tests.::

    pip install -r requirements-flake8.txt
    pip install -r requirements-extra.txt

10. Push changes in the branch to your forked repository on GitHub. ::

     git push origin feature-xyz

11. Create a merge request
