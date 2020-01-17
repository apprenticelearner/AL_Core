import logging, sys, os
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format='%(message)s | \'%(name)s:%(lineno)s\'')
log = logging.getLogger(os.path.basename(__file__))