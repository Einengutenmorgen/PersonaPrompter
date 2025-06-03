from abc import ABC, abstractmethod

class EvaluationResult(ABC):

    @abstractmethod
    def sorted(self, method='default'):
        """Get the results in the form of a sorted prompt and score list.
        Has a method argument to support sorting by various metrics."""
        pass

    @abstractmethod
    def in_place(self, method='default'):
        """Get the results in the form of a list of prompts and scores without sorting."""
        pass