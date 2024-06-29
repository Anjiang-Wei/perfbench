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

def main():
    parser = argparse.ArgumentParser(description="Run cholesky with specified parameters.")
    parser.add_argument("supercomputer", choices=["lassen", "sapling"], help="Supercomputer name")
    parser.add_argument("nodes", type=int, help="Number of nodes")
    parser.add_argument("size", type=int, help="Size of the matrix")
    parser.add_argument("num_partition", type=int, help="Number of partitions along each dimension")
    parser.add_argument("--spy", action='store_true', help="Turn on spy")

    args = parser.parse_args()

    numgpus = 4  # Set the number of GPUs to 4
    npieces = args.nodes * numgpus

    base_command = [
        "cholesky",
        f"-n {args.size}",
        f"-p {args.num_partition}",
        "-hl:sched 1024",
        f"-ll:gpu {numgpus}",
        "-ll:util 2",
        "-ll:bgwork 2",
        "-ll:csize 150000",
        "-ll:fsize 15000",
        "-ll:zsize 2048",
        "-ll:rsize 512",
        "-ll:gsize 0",
        f"-lg:prof {args.nodes}",
        f"-lg:prof_logfile prof_cholesky_{args.nodes}_%.gz",
    ]

    if args.spy:
        base_command.extend([
            "-lg:spy",
            f"-logfile spy_cholesky_{args.nodes}_%.log"
        ])

    command = get_header(args.supercomputer, args.nodes) + base_command

    env = os.environ.copy()
    execute_command(command, env)

if __name__ == "__main__":
    main()
