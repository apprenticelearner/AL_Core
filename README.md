This is a platform for creating apprentice learner agents.


# Installation
The main github repo is here: https://github.com/cmaclell/apprentice_learner_api

Clone the repo and then run:

pip install -r requirements.txt

That should install all of the required python modules.

Then you should be able to run:

python manage.py runserver

If you see a message that looks something like:

System check identified 1 issue (0 silenced).
September 19, 2018 - 13:33:31
Django version 1.10, using settings 'agent_api.settings'
Starting development server at http://127.0.0.1:8000/
Quit the server with CTRL-BREAK.

Then at least it to compiled. Kill that and run:

python manage.py createsuperuser

Follow the prompts to create a user account for yourself. Then run the server again. In a browser you should be able to go to http://127.0.0.1:8000/admin and log in. You should see the backend admin page for the service where you maybe? See some existing agents.

I almost certainly skipped some steps in there so chime in if you run into any problems.

# Papers and Documentation
1. Chris’s Thesis [link] specifically Chapter 3 is probably the best current source on the framework in writing
2. The Apprentice Learner Architecture EDM paper [link] is the best published reference, but the model diagram in this one is outdated
3. The TRESTLE Algorithm [link]
4. TRESTLE docs [link] contains The documentation for the actual TRESTLE software library, these are generally more accurate than the published papers but nothing has substantially changed (e.g., we technically calculate category utility differently now but in a way that produces the exact same behavior with less computation). This is also useful to check out because we use the preprocessors from this library throughout the apprentice to do things like flatten states for processing.
5. The Django Docs [link]
6. Py Search Docs [link] is the search library that is used throughout the apprentice/trestle

# Usage

# Extension
