#!/bin/bash
#SBATCH -A MST108253        # Account name/project number
#SBATCH -J get_tokens      # Job name
#SBATCH -p ct56             # Partiotion name
#SBATCH -n 8               # Number of MPI tasks (i.e. processes)
#SBATCH -c 1                # Number of cores per MPI task                         
#SBATCH -N 1                # Maximum number of nodes to be allocated
#SBATCH -o %j.out           # Path to the standard output file
#SBATCH -e %j.err           # Path to the standard error ouput file


python3 get_tokens.py
