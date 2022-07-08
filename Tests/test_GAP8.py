import os.path
import sys
import re
import subprocess
import pytest

sys.path.append('..')
from network_generate import network_generate


networks = [
    {
        "network_args":
            {
                'frontend': 'Quantlab',
                'target': 'GAP8.GAP8_gvsoc',
                'conf_file': '../dory_examples/config_files/config_Quantlab_MV1_4bits.json',
                'optional': 'mixed-sw'
            },
        "checksum_final": 'Failed [-7514 vs. -8000]'
    },
    {
        "network_args":
            {
                'frontend': 'Quantlab',
                'target': 'GAP8.GAP8_gvsoc',
                'conf_file': '../dory_examples/config_files/config_Quantlab_MV1_8bits.json',
                'optional': 'mixed-sw'
            },
        "checksum_final": 'Failed [-12656 vs. -13126]'
    },
    {
        "network_args":
            {
                'frontend': 'Quantlab',
                'target': 'GAP8.GAP8_gvsoc',
                'conf_file': '../dory_examples/config_files/config_Quantlab_MV2_4bits.json',
                'optional': 'mixed-sw'
            },
        "checksum_final": 'Failed [-10209 vs. -10509]'
    },
    {
        "network_args":
            {
                'frontend': 'Quantlab',
                'target': 'GAP8.GAP8_gvsoc',
                'conf_file': '../dory_examples/config_files/config_Quantlab_MV2_8bits.json',
                'optional': 'mixed-sw'
            },
        "checksum_final": 'Failed [5626 vs. 5297]'
    }
]


regex = re.compile(r'Checksum final :\s*(.*)$', re.MULTILINE)


def output_test(output, checksum_final):
    match = regex.search(output)
    return match.group(1) == checksum_final


@pytest.mark.parametrize('network', networks)
def test_network(network):
    args = network['network_args']
    checksum_final = network['checksum_final']

    network_generate(**args)

    cmd = ['make', '-C', 'application', 'clean', 'all', 'run', 'platform=gvsoc']
    try:
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=60)
    except subprocess.CalledProcessError as e:
        assert False, f"Building application failed with exit status {e.returncode}\nBuild error:\n{e.stderr}"

    network_name = os.path.splitext(os.path.basename(args['conf_file']))[0]
    preamble = f'Network {network_name}'

    assert output_test(proc.stdout, checksum_final), f'{preamble} Makefile output:\n {proc.stdout}'
