"""Utilities for managing SLURM experiments."""


def format_config_arg(key, val):
    """Format a configuration key-value pair as command line argument."""
    if val is None:
        command_arg = ""
    elif type(val) is bool:
        if val:
            command_arg = "--{}".format(key)
        else:
            command_arg = ""
    else:
        command_arg = "--{}={}".format(key, val)
    return command_arg


def update_job_file(job_file, job_name, output, array, command_file):
    """Update SLURM sbatch job file."""
    updated_job_lines = []
    with open(job_file, "r") as handle:
        job_lines = handle.readlines()
        for line in job_lines:
            if line.startswith("#SBATCH --job-name"):
                line = f"#SBATCH --job-name={job_name}\n"
            if line.startswith("#SBATCH --output"):
                line = f"#SBATCH --output={output}\n"
            if line.startswith("#SBATCH --array"):
                line = f"#SBATCH --array={array}\n"
            if line.startswith("PARAMS_FILE"):
                line = 'PARAMS_FILE="{}"\n'.format(command_file)
            updated_job_lines.append(line)

    with open(job_file, "w") as handle:
        for line in updated_job_lines:
            handle.write(line)
    print(f"SLURM job file updated at {job_file}")
