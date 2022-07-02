from abc import ABC, abstractmethod


class Accelerator(ABC):

    @property
    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def weights_ko_len(self, ko, dw):
        pass

    @abstractmethod
    def weights_ki_size(self, ki, ks, qw, dw):
        pass

    @abstractmethod
    def weights_size(self, ko, ki, ks, qw, dw):
        pass

    @abstractmethod
    def conv_unroll(self, w, qw, format='CoutCinK', dw=False):
        pass
