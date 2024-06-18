#!/usr/bin/env python3

import argparse
import subprocess
import os
import glob

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
    parser = argparse.ArgumentParser(description="Run circuit with specified parameters.")
    parser.add_argument("supercomputer", choices=["lassen", "sapling"], help="Supercomputer name")
    parser.add_argument("nodes", type=int, help="Number of nodes")
    parser.add_argument("--npp", type=int, required=True, help="Number of nodes per piece")
    parser.add_argument("--wpp", type=int, required=True, help="Number of wires per piece")
    parser.add_argument("--p", type=int, required=True, help="Number of pieces per (machine) node")
    parser.add_argument("--pps", type=int, required=True, help="Piece per superpiece")
    parser.add_argument("--noperf", action='store_true', help="Turn off performance profiling")
    parser.add_argument("--spy", action='store_true', help="Turn on spy profiling")
    parser.add_argument("--mapping", type=int, help="Specify mapping file number")

    args = parser.parse_args()

    if args.mapping is not None:
        mapping_files = [f"mapping{args.mapping}"]
    else:
        mapping_files = glob.glob("mapping[0-9]*")
    
    if not mapping_files:
        print("No mapping files found.")
        return

    for mapping_file in mapping_files:
        mapping_name = os.path.basename(mapping_file)
        
        base_command = [
            "circuit",
            f"-npp {args.npp}",
            f"-wpp {args.wpp}",
            "-l 10",
            f"-p {args.p * args.nodes}",
            f"-pps {args.pps}",
            "-prune 30",
            "-ll:util 2",
            "-ll:bgwork 2",
            "-hl:sched 1024",
            "-ll:gpu 4",
            "-ll:cpu 4",
            "-ll:csize 150000",
            "-ll:fsize 15000",
            "-ll:zsize 2048",
            "-ll:rsize 512",
            "-ll:gsize 0",
            f"-mapping {mapping_name}",
            "-wrapper",
            "-level mapper=2",
            f"-logfile wrapper_circuit_{mapping_name}_{args.nodes}_%.log",
        ]

        if not args.noperf:
            base_command.extend([
                f"-lg:prof {args.nodes}",
                f"-lg:prof_logfile prof_circuit_{mapping_name}_{args.nodes}_%.gz",
            ])

        if args.spy:
            base_command.extend([
                "-lg:spy",
                f"-logfile spy_circuit_{mapping_name}_{args.nodes}_%.log"
            ])

        command = get_header(args.supercomputer, args.nodes) + base_command

        env = os.environ.copy()
        execute_command(command, env)

if __name__ == "__main__":
    main()
