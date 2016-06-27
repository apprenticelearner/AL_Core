
from concept_formation.trestle import TrestleTree

from agents.BaseAgent import BaseAgent

class TrestleHow(BaseAgent):
   """
   This is an agent based on a single Trestle knowledge base contianing both
   the contexts of actions and the explained parameters of actions. It is
   based on the original implementation of the trestle_api.
   """

   def __init__(self,how_params=None):
        self.tree = TrestleTree()
        self.how_params = how_params