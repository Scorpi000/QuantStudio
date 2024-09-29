from abc import abstractmethod, ABC


class CacheFun(ABC):
    @abstractmethod
    def write_cache(self):
        pass

    @abstractmethod
    def read_cache(self, key):
        pass

    @abstractmethod
    def clear_cache(self, key):
        pass

    @abstractmethod
    def check_ready(self):
        pass
