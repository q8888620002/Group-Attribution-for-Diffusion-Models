import os
import glob

if __name__ == "__main__":

    seeds = [seed for seed in range(1000)]

    base_model_outdir = "/gscratch/scrubbed/mingyulu/diffusion-attr/cifar/gd/models/datamodel"

    for seed in seeds:
        model_outdir = f"{base_model_outdir}/datamodel_alpha=0.5_seed={seed}"
        pattern = os.path.join(model_outdir, "ckpt_steps_*.pt")
        max_step = -1
        max_step_file = ""

        for filename in glob.glob(pattern):
            step_num = int(filename.split('_')[-1].split('.')[0])  # Extract step number from the filename
            if step_num > max_step:
                max_step = step_num
                max_step_file = filename

        if max_step_file:
            for filename in glob.glob(pattern):
                if filename != max_step_file:
                    os.remove(filename)
            print(f"Kept max step file for seed {seed}: {max_step_file} and removed others.")
        else:
            print(f"No checkpoint files found for seed {seed}.")
