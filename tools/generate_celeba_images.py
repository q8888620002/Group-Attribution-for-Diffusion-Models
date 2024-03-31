import itertools
import os
from diffusers import StableDiffusionPipeline
import src.constants as constants

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
    "White"
]

gender = [
    "female",
    "male",
    "non-binary"
]

# Generate all combinations of ethnicity and gender
combinations = list(itertools.product(ethnicity, gender))

all_prompts = []

pipe = StableDiffusionPipeline.from_pretrained("lambdalabs/miniSD-diffusers")
pipe = pipe.to("cuda")
pipe.safety_checker = None

counter = 1

for combo in combinations:
    ethnicity, gender = combo
    for _ in range(40):  # Repeat 40 times for even distribution
        prompt = f"a full-color photograph of a single, {gender} and {ethnicity} celebrity."
        all_prompts.append(prompt)
        image = pipe(prompt, width=256, height=256).images[0]  
        image.save(os.path.join('results/celeba', f'test_{counter}.jpg'))
        counter += 1