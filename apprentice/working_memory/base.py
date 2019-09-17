from abc import ABCMeta, abstractmethod


class WorkingMemory(metaclass=ABCMeta):
    @abstractmethod
    def get_facts(self):
        pass

    @abstractmethod
    def get_skills(self):
        pass

    @abstractmethod
    def add_fact(self, fact):
        pass

    @abstractmethod
    def add_skill(self, skill):
        pass

    @abstractmethod
    def update_fact(self, fact):
        pass

    @abstractmethod
    def update_skill(self, skill):
        pass
