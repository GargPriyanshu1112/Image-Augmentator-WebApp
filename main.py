import streamlit as st
from utils import (
    get_augmentations,
    get_image_transformer,
    get_zipfile,
    is_num_augs_valid,
)


def get_state(state, status):
    if status is None:
        return st.session_state[state]
    return status


def update_state(state, status):
    st.session_state[state] = status


if "is_aug_type_selection_done" not in st.session_state:
    st.session_state["is_aug_type_selection_done"] = False


st.title("Image Augmentator")

files_uploaded = st.file_uploader(
    label="Select image(s)...", type=["jpg", "png"], accept_multiple_files=True
)


with st.container(border=True):
    st.write("Select the augmentation types you want to apply.")

    col_1, col_2, col_3 = st.columns(3)
    col_4, col_5, col_6 = st.columns(3)

    checkbox_Rotation = col_1.checkbox(
        label="Rotation",
        help="Rotates the input by an angle selected randomly from the uniform distribution.",
        disabled=not files_uploaded,
    )

    checkbox_HFlip = col_2.checkbox(
        label="Horizontal Flip",
        help="Flip the input horizontally around the y-axis.",
        disabled=not files_uploaded,
    )

    checkbox_VFlip = col_3.checkbox(
        label="Vertical Flip",
        help="Flip the input vertically around the x-axis.",
        disabled=not files_uploaded,
    )

    checkbox_Zoom = col_4.checkbox(
        label="Random Resized Crop",
        help="Crops a random part of the input and rescale it to some size.",
        disabled=not files_uploaded,
    )

    checkbox_ShiftRGB = col_5.checkbox(
        label="RGB Shift",
        help="Randomly shift values for each channel of the input RGB image.",
        disabled=not files_uploaded,
    )

    checkbox_ColorJitter = col_6.checkbox(
        label="Color Jitter",
        help="Randomly changes the brightness, contrast, and saturation of an image.",
        disabled=not files_uploaded,
    )

    num_augs_selected = [
        checkbox_Rotation,
        checkbox_HFlip,
        checkbox_VFlip,
        checkbox_Zoom,
        checkbox_ShiftRGB,
        checkbox_ColorJitter,
    ].count(True)

    is_aug_types_selected = st.button(
        label="Continue",
        on_click=update_state("is_aug_type_selection_done", True),
        disabled=(not files_uploaded) or (num_augs_selected == 0),
    )

    num_augs = st.text_input(
        label="No. of augmentated images per image (0 - 100): ",
        disabled=not is_aug_types_selected,
    )

    info_entered = st.button(
        label="Done",
        disabled=not is_num_augs_valid(num_augs),
    )


with st.container():
    if st.button("Download Zip File", disabled=not info_entered):
        transformer = get_image_transformer(
            checkbox_Rotation,
            checkbox_HFlip,
            checkbox_VFlip,
            checkbox_Zoom,
            checkbox_ShiftRGB,
            checkbox_ColorJitter,
        )
        augmentations = get_augmentations(transformer, files_uploaded, int(num_augs))
        get_zipfile(augmentations)
        st.success("Downloaded !")
