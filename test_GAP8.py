import os.path
import sys
import re
import subprocess

# conftest.py
import pytest
from _pytest.capture import capsys

from network_generate import network_generate


networks = [
    {
            "network_args":
            {
                'frontend': 'Quantlab',
                'target': 'GAP8.GAP8_gvsoc',
                'conf_file': './dory/dory_examples/config_files/config_Quantlab_MV1_fast_xpnn.json',
                'optional': 'mixed-hw'
            },
            "checksum_final": 'Ok'
    },
    {
        "network_args":
            {
                'frontend': 'NEMO',
                'target': 'GAP8.GAP8_board',
                'conf_file': './dory/dory_examples/config_files/config_NEMO_Penguinet_64.json'
            },
        "checksum_final": 'Ok'
    },
    {
        "network_args":
            {
                'frontend': 'NEMO',
                'target': 'GAP8.GAP8_board',
                'conf_file': './dory/dory_examples/config_files/config_NEMO_dronet.json'
            },
        "checksum_final": 'Ok'
    },
    {
        "network_args":
            {
                'frontend': 'Quantlab',
                'target': 'GAP8.GAP8_gvsoc',
                'conf_file': './dory/dory_examples/config_files/config_Quantlab_MV1_4bits.json',
                'optional': 'mixed-sw'
            },
        "checksum_final": 'Ok'
    },
    {
        "network_args":
            {
                'frontend': 'Quantlab',
                'target': 'GAP8.GAP8_gvsoc',
                'conf_file': './dory/dory_examples/config_files/config_Quantlab_MV1_8bits.json',
                'optional': 'mixed-sw'
            },
        "checksum_final": 'Ok',
        "compat": ["gap-sdk"]
    },
    {
        "network_args":
            {
                'frontend': 'Quantlab',
                'target': 'GAP8.GAP8_gvsoc',
                'conf_file': './dory/dory_examples/config_files/config_Quantlab_MV2_4bits.json',
                'optional': 'mixed-sw'
            },
        "checksum_final": 'Ok'
    },
    {
        "network_args":
            {
                'frontend': 'Quantlab',
                'target': 'GAP8.GAP8_gvsoc',
                'conf_file': './dory/dory_examples/config_files/config_Quantlab_MV2_8bits.json',
                'optional': 'mixed-sw'
            },
        "checksum_final": 'Ok'
    },
    {
        "network_args":
            {
                'frontend': 'Quantlab',
                'target': 'GAP8.GAP8_board',
                'conf_file': './dory/dory_examples/config_files/config_Quantlab_MV1_8bits.json',
                'optional': 'mixed-sw'
            },
        "checksum_final": 'Ok'
    }
]



regex = re.compile(r'Checksum final :\s*(.*)$', re.MULTILINE)
regex_cycles = re.compile(r'num_cycles:\s*(.*)$', re.MULTILINE)
regex_MACs = re.compile(r'MACs:\s*(.*)$', re.MULTILINE)

def output_test(output, checksum_final):
    match = regex.search(output)
    return match.group(1) == checksum_final


# check if a network is compatible with the specified SDK (must be gap-sdk or
# pulp-sdk)
def check_compat(network : dict, compat : str):
    try:
        compat_sdks = network["compat"]
    except KeyError:
        compat_sdks = ["gap-sdk", "pulp-sdk"]

    net_args = network["network_args"]

    if compat == 'pulp-sdk':
        # pulp-sdk can handle everything
        return compat in compat_sdks

    try:
        optional = net_args['optional']
    except KeyError:
        optional = 'auto'

    return (not optional == 'mixed-hw') and (compat in compat_sdks)


@pytest.mark.parametrize('network', networks)
def test_network(network, capsys, compat):
    args = network['network_args']
    if not check_compat(network, compat):
        with capsys.disabled():
            print(f"Skipping network with conf_file {args['conf_file']} as it is not compatible with SDK {compat}")
        return
    checksum_final = network['checksum_final']
    network_generate(**args)

    cmd = ['make', '-C', 'application', 'clean', 'all', 'run', 'platform=gvsoc', 'CORE=8']
    try:
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=180)
    except subprocess.CalledProcessError as e:
        assert False, f"Building application failed with exit status {e.returncode}\nBuild error:\n{e.stderr}"
    except subprocess.TimeoutExpired as e:
        print(f"Test timed out...\nSTDOUT:")
        if e.output is not None:
            print(e.output.decode())
        print(f"STDERR:")
        if e.stderr is not None:
            print(e.stderr.decode())
        exit(1)

    network_name = os.path.splitext(os.path.basename(args['conf_file']))[0]
    preamble = f'Network {network_name}'
    with capsys.disabled():
        print('')
    if output_test(proc.stdout, checksum_final):
        cycl = regex_cycles.search(proc.stdout).group(1)
        MACs = regex_MACs.search(proc.stdout).group(1)
        with capsys.disabled():
            print(f'{preamble.ljust(40)}, MACs: {MACs.rjust(10)}, Cycles: {cycl.rjust(10)}, MAC/cycles: {(int(MACs)/float(cycl)):.4f}')
    else:
        exit(1)
        with capsys.disabled():
            print(f'{preamble.ljust(40)}, Failed')
        #print(f'{preamble} Makefile output:\n {proc.stdout}')
