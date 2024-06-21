#!/usr/bin/env python3

import argparse
import subprocess
import os
import math

def execute_command(command, env):
    command_str = " ".join(command)
    print(f"Executing command: {command_str}", flush=True)
    
    result = subprocess.run(command_str, shell=True, capture_output=True, env=env)
    
    print(result.stdout.decode(), flush=True)
    print(result.stderr.decode(), flush=True)

def get_header(supercomputer, nodes):
    if supercomputer == "lassen":
        return ["jsrun", "-b", "none", "-c", "ALL_CPUS", "-g", "ALL_GPUS", "-r", "1", "-n", f"{nodes}"]
    else:
        return ["mpirun", "--bind-to", "none"]

def lgGPUArgs(gpus):
    return [
        '-ll:ocpu', '1',
        '-ll:othr', '10',
        '-ll:csize', '150000',
        '-ll:util', '4',
        '-dm:replicate', '1',
        '-ll:gpu', str(gpus),
        '-ll:fsize', '15000',
        '-ll:bgwork', '12',
        '-ll:bgnumapin', '1',
    ]

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

def backpressureArgs(procs):
    if procs in [8, 32, 128]:
        return ['-tm:enable_backpressure', '-tm:backpressure_max_in_flight', '1', '-ll:defalloc', '0']
    else:
        return []

def get_cannon_gpu_command(supercomputer, nodes, gpus, size, wrapper, prof, spy):
    psize = weak_scaling_size(size, nodes)
    gx = nearest_square(nodes)
    gy = nodes // gx
    header = get_header(supercomputer, nodes)
    base_command = [
        'main', '-n', str(psize), '-gx', str(gx), '-gy', str(gy),
        '-dm:exact_region', '-tm:untrack_valid_regions'
    ] + lgGPUArgs(gpus) + backpressureArgs(nodes)

    if wrapper:
        base_command = [
            '-wrapper', '-level', 'mapper=debug', '-logfile', f"wrapper_cannon-cuda_{nodes}_%.log"
        ] + base_command
    
    if prof:
        base_command += ['-lg:prof', str(nodes), '-lg:prof_logfile', f'prof_cannon-cuda_{nodes}_%.gz']

    if spy:
        base_command += ['-lg:spy', '-logfile', f'spy_cannon-cuda_{nodes}_%.log']

    return header + base_command

def main():
    parser = argparse.ArgumentParser(description="Run cannonGPU benchmark with specified parameters.")
    parser.add_argument("supercomputer", choices=["lassen", "sapling"], help="Supercomputer name")
    parser.add_argument("nodes", type=int, help="Number of nodes")
    parser.add_argument("--size", type=int, required=True, help="Initial problem size for the benchmark")
    parser.add_argument("--gpus", type=int, default=4, help="Number of GPUs for GPU benchmarks")
    parser.add_argument("--wrapper", action='store_true', help="Enable wrapper command")
    parser.add_argument("--no-prof", action='store_false', help="Disable performance profiling", dest='prof')
    parser.add_argument("--spy", action='store_true', help="Enable spy logging")

    args = parser.parse_args()

    command = get_cannon_gpu_command(args.supercomputer, args.nodes, args.gpus, args.size, args.wrapper, args.prof, args.spy)

    env = os.environ.copy()
    execute_command(command, env)

if __name__ == "__main__":
    main()
