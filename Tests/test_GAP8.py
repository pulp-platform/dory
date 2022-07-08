import sys
import pytest

sys.path.append('..')
from network_generate import network_generate

networks = [
    {
        'frontend': 'Quantlab',
        'target': 'GAP8.GAP8_gvsoc',
        'conf': '../dory_examples/config_files/config_Quantlab_MV1_4bits.json',
        'optional': 'mixed-sw'
    }
]


@pytest.mark.parametrize('network', networks)
def test_network(network):
    network_generate(**network)
