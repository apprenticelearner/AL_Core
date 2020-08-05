# Apprentice Learner Architecture

[![Build Status](https://travis-ci.org/apprenticelearner/AL_Core.svg?branch=master)](https://travis-ci.org/apprenticelearner/AL_Core) 
[![Coverage Status](https://coveralls.io/repos/github/apprenticelearner/AL_Core/badge.svg?branch=master)](https://coveralls.io/github/apprenticelearner/AL_Core?branch=master)
[![Documentation Status](https://readthedocs.org/projects/al-core/badge/?version=latest)](https://al-core.readthedocs.io/en/latest/?badge=latest)

The Apprentice Learner Architecture provides a framework for modeling and simulating learners working educational technologies. There are three general GitHub repositories for the AL Project: 

1. **AL_Core** (this repository), which is the core library for learner modeling used to configure and instantiate agents and author their background knowledge. 
2. **AL_Train** (https://github.com/apprenticelearner/AL_Train), which contains code for interfacing AL agents with CTAT-HTML tutors and running training experiments.
3. **AL_Outerloop** (https://github.com/apprenticelearner/AL_Outloop), which provides additional functionality to AL_Train simulating adaptive curricula.

# Installation

To install the AL_Core library, [clone both respositories](https://help.github.com/en/articles/cloning-a-repository) to your machine using the GitHub deskptop application or by running the following commands in a terminal / command line:

```bash
git clone https://github.com/apprenticelearner/AL_Core 
```

Navigate to the directory where you cloned AL_Core in a terminal / command line and run:

```bash
python -m pip install -e .
```

Next, go to the [pytorch setup guide](https://pytorch.org/get-started/locally/) and follow the sets specified for your operating system and environment to install a final depending.

Finally, change directory to AL_Core/django and run the migrations for the django configuration:

```bash
cd AL_Core/django/
python manage.py migrate
```

Everything should now be fully installed and ready.

# Important Links

* Source code: https://github.com/apprenticelearner/AL_Core
* Documentation: https://al-core.readthedocs.io/en/latest/

# Examples
We have created a number of examples to demonstrate basic usage of the Appentice Learner that make use of this repository as well as [AL_Train](https://github.com/apprenticelearner/AL_Core). These can be found on the [examples page](https://github.com/apprenticelearner/AL_Core/wiki/Examples) of the wiki.

# Citing this Software

If you use this software in a scientific publiction, then we would appreciate citation of the following paper:

Christopher J MacLellan, Erik Harpstead, Rony Patel, and Kenneth R Koedinger. 2016. The Apprentice Learner Architecture: Closing the loop between learning theory and educational data. In Proceedings of the 9th International Conference on Educational Data Mining - EDM ’16, 151–158. Retrieved from http://www.educationaldatamining.org/EDM2016/proceedings/paper_118.pdf

Bibtex entry:

```
@inproceedings{MacLellan2016a,
author = {MacLellan, Christopher J and Harpstead, Erik and Patel, Rony and Koedinger, Kenneth R},
booktitle = {Proceedings of the 9th International Conference on Educational Data Mining - EDM '16},
file = {:C$\backslash$:/Users/eharpste/Documents/Articles/MacLellan et al. - 2016 - The Apprentice Learner Architecture Closing the loop between learning theory and educational data.pdf:pdf},
pages = {151--158},
title = {{The Apprentice Learner Architecture: Closing the loop between learning theory and educational data}},
url = {http://www.educationaldatamining.org/EDM2016/proceedings/paper{\_}118.pdf},
year = {2016}
}
```
