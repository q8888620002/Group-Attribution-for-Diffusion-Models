"""Script that generate diversed celebrty images from miniSD."""
import itertools
import os

import src.constants as constants
from diffusers import StableDiffusionPipeline

ethnicity = [
    "African-American",
    "American Indian",
    "Asian",
    "Black",
    "Caucasian",
    "East Asian",
    "First Nations",
    "Hispanic",
    "Indigenous American",
    "Latino",
    "Latinx",
    "Native American",
    "Multiracial",
    "Pacific Islander",
    "South Asian",
    "Southeast Asian",
    "White",
]

gender = ["female", "male", "non-binary"]
age = ["adult", "senior"]

# Generate all combinations of ethnicity and gender
combinations = list(itertools.product(ethnicity, gender, age))

all_prompts = []

pipe = StableDiffusionPipeline.from_pretrained("lambdalabs/miniSD-diffusers")
pipe = pipe.to("cuda")
pipe.safety_checker = None

counter = 1

for combo in combinations:
    ethnicity, gender, age = combo

    for _ in range(40):  # Repeat 40 times for even distribution
        prompt = (
            f"a full-color, and high-resolution headshot of a single, "
            f"{age}, {gender} and {ethnicity} celebrity."
        )
        all_prompts.append(prompt)
        image = pipe(prompt, width=256, height=256).images[0]

        output_dir = os.path.join(constants.OUTDIR, "celeba/generated_samples")
        os.makedirs(output_dir, exist_ok=True)
        image.save(os.path.join(output_dir, f"celeba_{counter}.jpg"))
        counter += 1

print(f"Image generation done! Saved at {output_dir}.")