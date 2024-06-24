#!/usr/bin/env python3

import argparse
import subprocess
import os
import glob
import re

def execute_command(command, env):
    command_str = " ".join(command)
    print(f"Executing command: {command_str}", flush=True)
    
    result = subprocess.run(command_str, shell=True, capture_output=True, env=env)
    
    print(result.stdout.decode(), flush=True)
    print(result.stderr.decode(), flush=True)

def get_header(supercomputer, nodes):
    if supercomputer == "lassen":
        return ["jsrun", "-b", "none", "-c", "ALL_CPUS", "-g", "ALL_GPUS", "-r", "1", "-n", f"{nodes}"]
    elif supercomputer  == "sapling":
        return ["mpirun", "--bind-to", "none"]
    else:
        raise ValueError(f"Unknown supercomputer: {supercomputer}")

def lgCPUArgs(supercomputer, othrs=18):
    if supercomputer == "lassen":
        args = [
            '-ll:ocpu', '2',
            '-ll:othr', str(othrs),
            '-ll:onuma', '1',
            '-ll:csize', '5000',
            '-ll:nsize', '75000',
            '-ll:ncsize', '0',
            '-ll:util', '2',
            '-dm:replicate', '1',
        ]
        if (othrs != 18):
            args += ['-ll:ht_sharing', '0']
        return args
    elif supercomputer == "sapling":
        return [
            '-ll:ocpu', '2',
            '-ll:othr', '1',
            '-ll:csize', '5000',
            '-ll:nsize', '75000',
            '-ll:ncsize', '0',
            '-ll:util', '1',
            '-dm:replicate', '1',
        ]
    else:
        raise ValueError(f"Unknown supercomputer: {supercomputer}")

def nearest_square(max_val):
    val = 1
    while True:
        sq = val * val
        if sq > max_val:
            return val - 1
        if sq == max_val:
            return val
        val += 1

def weak_scaling_size(initial_size, procs, scaling_factor=1.0/2.0):
    size = int(initial_size * pow(procs, scaling_factor))
    size -= (size % 2)
    return size


def get_cannon_cpu_command(supercomputer, nodes, size, wrapper, prof, spy, taco, mapping_file):
    psize = weak_scaling_size(size, nodes)
    gx = nearest_square(nodes)
    gy = nodes // gx
    header = get_header(supercomputer, nodes)
    base_command = [
        'main', '-n', str(psize), '-gx', str(gx), '-gy', str(gy),
        '-dm:exact_region', '-tm:untrack_valid_regions'
    ] + lgCPUArgs(supercomputer)

    benchname = f'cannon-cpu_{mapping_file}' if not taco else 'cannon-cpu_taco'

    if wrapper:
        base_command = [
            '-wrapper', '-level', 'mapper=debug', '-logfile', f"wrapper_{benchname}_{nodes}_%.log"
        ] + base_command
    
    if prof:
        base_command += ['-lg:prof', str(nodes), '-lg:prof_logfile', f'prof_{benchname}_{nodes}_%.gz']

    if spy:
        base_command += ['-lg:spy', '-logfile', f'spy_{benchname}_{nodes}_%.log']

    if taco:
        base_command += ['-ll:show_rsrv']
    else: # use DSL mapper
        base_command += ['-dslmapper', '-mapping', f'{mapping_file}']

    return header + base_command

def sort_mapping_files(mapping_files):
    """
    Sort mapping files based on the numeric index in their names.
    """
    return sorted(mapping_files, key=lambda x: int(re.search(r'\d+', x).group()))

def main():
    parser = argparse.ArgumentParser(description="Run cannon benchmark with specified parameters.")
    parser.add_argument("supercomputer", choices=["lassen", "sapling"], help="Supercomputer name")
    parser.add_argument("nodes", type=int, help="Number of nodes")
    parser.add_argument("--size", type=int, required=True, help="Initial problem size for the benchmark")
    parser.add_argument("--wrapper", action='store_true', help="Enable wrapper command")
    parser.add_argument("--no-prof", action='store_false', help="Disable performance profiling", dest='prof')
    parser.add_argument("--spy", action='store_true', help="Enable spy logging")
    parser.add_argument("--taco", action='store_true', help="Enable original taco mapper")
    parser.add_argument("--mapping", type=int, help="Specify mapping file number")

    args = parser.parse_args()
    if args.mapping is not None:
        mapping_files = [f"mapping{args.mapping}"]
    else:
        mapping_files = glob.glob("mapping[0-9]*")
        mapping_files = sort_mapping_files(mapping_files)
    
    if len(mapping_files) == 0:
        print("No mapping files found.")
        return
    
    for mapping_file in mapping_files:
        command = get_cannon_cpu_command(args.supercomputer, args.nodes, args.size, args.wrapper, args.prof, args.spy, args.taco, mapping_file)

        env = os.environ.copy()
        execute_command(command, env)

if __name__ == "__main__":
    main()
