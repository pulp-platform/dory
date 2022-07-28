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
    def conv_unroll(self, w, qw, layout='CoutCinK', dw=False):
        pass

    @abstractmethod
    def heuristic_l1(self, n_out, n_in, h_out, w_out,
                     tile_n_out, tile_n_in, tile_h_out, tile_w_out,
                     constr_total_size, ks, modifier):
        pass

    @abstractmethod
    def heuristic_l2(self, tile_n_out, tile_n_in, tile_h_out,
                     constr_total_size, ks, modifier):
        pass
