import logging
import logging.config
import os

# import so "disable_loggers" can have effect
#from experta import unwatch

#unwatch()

import coloredlogs
import yaml

from . import working_memory, agents, learners, planners


def setup_logging(default_path='logging.yaml', default_level=logging.INFO,
                  env_key='LOG_CFG'):
    # https://gist.github.com/kingspp/9451566a5555fb022215ca2b7b802f19
    path = default_path
    value = os.getenv(env_key, None)
    if value:
        path = value

    if os.path.exists(path):
        with open(path, 'rt') as f:
            try:
                config = yaml.safe_load(f.read())
                logging.config.dictConfig(config)
            except Exception as e:
                print(e)
                print('Error in Logging Configuration. Using default configs')
                logging.basicConfig(level=default_level)
                coloredlogs.install(level=default_level)
    else:
        logging.basicConfig(level=default_level)
        coloredlogs.install(level=default_level)
        print('Failed to load configuration file. Using default configs')


log_config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                               'logging.yaml')

setup_logging(default_path=log_config_path)
# '%(name)s:%(lineno)s | %(message)s'
