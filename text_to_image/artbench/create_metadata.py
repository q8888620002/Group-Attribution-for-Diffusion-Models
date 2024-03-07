"""
Create the metadata.csv file for the ArtBench-10 dataset. This is to allow loading with
HuggingFace datasets.load_dataset().
https://huggingface.co/docs/datasets/v2.17.0/en/package_reference/loading_methods#datasets.load_dataset
"""


import argparse
import os
import re

import pandas as pd

from src.ddpm_config import PromptConfig

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="create ArtBench-10 metdata.csv")
    parser.add_argument(
        "--parent_dir",
        help="parent directory containing the the train and test subdirectories",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--split",
        help="train or test split",
        choices=["train", "test"],
        type=str,
        required=True,
    )
    parser.add_argument(
        "--diverse",
        help="whether to generate diverse captions",
        action="store_true",
    )
    args = parser.parse_args()

    caption_dict = PromptConfig.artbench_config

    data_dir = os.path.join(args.parent_dir, args.split)
    print(f"Creating metadata for {data_dir}")

    art_styles = [
        item
        for item in os.listdir(data_dir)
        if not item.startswith(".") and not item.endswith(".csv")
    ]
    df_list = []
    art_style_captions = []
    for art_style in art_styles:
        img_files, artists, captions = [], [], []
        for img_file in os.listdir(os.path.join(data_dir, art_style)):
            img_files.append(os.path.join(art_style, img_file))
            artist = img_file.split("_")[0]
            artists.append(artist)

            if args.diverse:
                formatted_artist = artist.replace("-", " ")
                formatted_artist = formatted_artist.title()

                # Handle the suffixes II, III, etc.
                formatted_artist = re.sub(
                    " i+$", lambda x: x[0].upper(), formatted_artist
                )

                title = img_file.replace(".jpg", "").split("_")[1]
                title = title.replace("-", " ").title()

                caption = (
                    title + ", " + caption_dict[art_style] + " by " + formatted_artist
                )
            else:
                caption = caption_dict[art_style]
            captions.append(caption)

        art_style_captions.append(caption_dict[art_style])
        art_style_df = pd.DataFrame(
            {
                "file_name": img_files,
                "caption": captions,
                "artist": artists,
                "style": art_style,
                "filename": img_files,  # Duplicate for access in Hugging Face dataset.
            }
        )
        unique_artists = art_style_df["artist"].unique().tolist()
        unique_artists = sorted(unique_artists)

        artists_outfile = os.path.join(data_dir, f"{art_style}_artists.csv")
        pd.DataFrame({"artist": unique_artists}).to_csv(artists_outfile, index=False)
        print(f"{art_style} artist names saved to {artists_outfile}")

        file_names_outfile = os.path.join(data_dir, f"{art_style}_filenames.csv")
        pd.DataFrame({"filename": img_files}).to_csv(file_names_outfile, index=False)
        print(f"{art_style} image file names saved to {file_names_outfile}")

        df_list.append(art_style_df)
    df = pd.concat(df_list)

    # Check known statistics.
    if args.split == "train":
        num_imgs = 50000
        num_imgs_per_style = 5000
    else:
        num_imgs = 10000
        num_imgs_per_style = 1000
    assert len(df) == num_imgs
    print("Captions are")
    for art_style in art_styles:
        assert len(df[df["style"] == art_style]) == num_imgs_per_style
        print(f"\t{caption_dict[art_style]}")

    outfile = os.path.join(data_dir, "metadata.csv")
    df.to_csv(outfile, index=False)
    print(f"Metadata saved to {outfile}")
