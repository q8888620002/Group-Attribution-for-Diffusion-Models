# Explaining Diffusion Models via Sparsified Unlearning

## Overview
With the widespread usage of diffusion models, effective data attribution is needed
to ensure fair acknowledgment for contributors of high-quality training samples,
and to identify potential sources of harmful content. In this early work, we in-
troduce a novel framework tailored to removal-based data attribution for diffusion
models, leveraging sparsified unlearning. This approach significantly improves
the computational scalability and effectiveness of removal-based data attribution.
In our experiments, we attribute diffusion model FID back to CIFAR-10 train-
ing images with datamodel attributions, showing better linear datamodeling score
(LDS) than datamodel attributions based on naive retraining

## Directory Structure

```plaintext
.
├── src/  # Files for common Python functions and classes.
│   ├── ddpm_config.py
│   ├── utils.py
│   └── # some other module files
│
├── unconditional_generation/  # Files for unconditional diffusion models.
│   ├── README.md
│   ├── main.py
│   ├── prune.py
│   ├── generate_samples.py
│   ├── calculate_local_scores.py
│   ├── cifar/
│   │   ├── calculate_global_scores.py  # Global scores can differ between different datasets.
│   │   └── results.ipynb  # Jupyter notebook(s) for plotting results.
│   ├── celeba/
│   │   ├── calculate_global_scores.py
│   │   └── results.ipynb
│   └── experiments/  # Files for managing SLURM experiments.
│
├── text_to_image/  # Files for text-to-image diffusion models.
│   ├── README.md
│   ├── main.py
│   ├── prune.py
│   ├── generate_samples.py
│   ├── calculate_local_scores.py
│   ├── artbench/
│   │   ├── calculate_global_scores.py
│   │   └── results.ipynb
│   └── experiments/  # Files for managing SLURM experiments.
│
├── some_common_script_0.py  # Script(s) that are useful for all diffusion models.
├── some_common_script_1.py

## Installation
Provide instructions on how to install and run the project.

## Usage
Describe how to use the project, including any relevant commands.

## Contributing
Guidelines for contributing to the project, if applicable.

## License
Specify the license under which the project is released.
```

Feel free to fill in the sections with appropriate details specific to your project, such as an overview, installation instructions, usage examples, contributing guidelines, and licensing information.
