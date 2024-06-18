#!/usr/bin/env python3

import argparse
import subprocess
import os
import re
import glob
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

def factorize_nodes(nodes):
    """ 
    Factorize nodes into two factors as close to each other as possible.
    """
    root = int(math.sqrt(nodes))
    for i in range(root, 0, -1):
        if nodes % i == 0:
            return i, nodes // i
    return 1, nodes

def sort_mapping_files(mapping_files):
    """
    Sort mapping files based on the numeric index in their names.
    """
    return sorted(mapping_files, key=lambda x: int(re.search(r'\d+', x).group()))

def main():
    parser = argparse.ArgumentParser(description="Run stencil with specified parameters.")
    parser.add_argument("supercomputer", choices=["lassen", "sapling"], help="Supercomputer name")
    parser.add_argument("nodes", type=int, help="Number of nodes")
    parser.add_argument("--tile_size", type=int, required=True, help="Tile size multiplier")
    parser.add_argument("--noperf", action='store_true', help="Turn off performance profiling")
    parser.add_argument("--spy", action='store_true', help="Turn on spy profiling")
    parser.add_argument("--mapping", type=int, help="Specify mapping file number")

    args = parser.parse_args()

    nx, ny = factorize_nodes(args.nodes)

    if args.mapping is not None:
        mapping_files = [f"mapping{args.mapping}"]
    else:
        mapping_files = glob.glob("mapping[0-9]*")
        mapping_files = sort_mapping_files(mapping_files)
    
    if not mapping_files:
        print("No mapping files found.")
        return

    for mapping_file in mapping_files:
        mapping_name = os.path.basename(mapping_file)
        
        base_command = [
            "stencil",
            f"-nx {args.tile_size * nx}",
            f"-ny {args.tile_size * ny}",
            f"-ntx {nx * 2}",
            f"-nty {ny * 2}",
            "-tsteps 10",
            "-tprune 30",
            "-hl:sched 1024",
            "-ll:gpu 4",
            "-ll:util 1",
            "-ll:bgwork 2",
            "-ll:csize 150000",
            "-ll:fsize 15000",
            "-ll:zsize 2048",
            "-ll:rsize 512",
            "-ll:gsize 0",
            f"-mapping {mapping_name}",
            "-wrapper",
            f"-logfile wrapper_stencil_{mapping_name}_{args.nodes}_%.log",
        ]

        if not args.noperf:
            base_command.extend([
                f"-lg:prof {args.nodes}",
                f"-lg:prof_logfile prof_stencil_{mapping_name}_{args.nodes}_%.gz",
            ])

        if args.spy:
            base_command.extend([
                "-lg:spy",
                f"-logfile spy_stencil_{mapping_name}_{args.nodes}_%.log"
            ])

        command = get_header(args.supercomputer, args.nodes) + base_command

        env = os.environ.copy()
        execute_command(command, env)

if __name__ == "__main__":
    main()
