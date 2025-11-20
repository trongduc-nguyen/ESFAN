We release a weakly-supervised tissue semantic segmentation dataset for breast cancer named BCSS-WSSS, which is generated from Breast Cancer Semantic Segmentation (BCSS) Dataset.[1]
BCSS-WSSS dataset includes four tissue categories, Tumor (TUM), Stroma (STR), Lymphocytic infiltrate (LYM) and Necrosis (NEC). The details of this dataset are shown below:

# Folder Structure

BCSS-WSSS/
    |_train/                                                                    * Training set with patch-level annotations
    |   |_TCGA-A1-A0SK-DX1_xmin45749_ymin25055_MPP-0.2500+0[1101].png           * Patches cropped from images in public BCSS dataset. 'Image-name-of-BCSS'+'+index'+'[abcd]'.png
    |   |_......                                                                * The total number of patches in the training set is 23,422
    |
    |_val/                                                                      * Validation set with pixel-level annotations
    |   |_img/
    |   |   |_TCGA-EW-A1PB-DX1_xmin57214_ymin25940_MPP-0.2500+0.png             * Patches cropped from images in public BCSS dataset. 'Image-name-of-BCSS'+'+index'+.png
    |   |   |_......                                                            * The total number of patches in the validation set is 3,418
    |   |_mask/
    |   |   |_TCGA-EW-A1PB-DX1_xmin57214_ymin25940_MPP-0.2500+0.png             * The mask share the same filename with the original patch.
    |   |   |_......                                                            * The total number of masks in the validation set is 3,418
    |
    |_test/
    |   |_img/
    |   |   |_TCGA-D8-A27F-DX1_xmin98787_ymin6725_MPP-0.2500+0.png              * Patches cropped from images in public BCSS dataset. 'Image-name-of-BCSS'+'+index'+.png
    |   |   |_......                                                            * The total number of patches in the test set is 4,986
    |   |_mask/
    |   |   |_TCGA-D8-A27F-DX1_xmin98787_ymin6725_MPP-0.2500+0.png              * The mask share the same filename with the original patch.
    |   |   |_......                                                            * The total number of masks in the test set is 4,986
    |
    |_Readme.txt                                                                * Readme file of this dataset
    |_Example_for_using_image-level_label.py                                    * Source code to read the patch-level labels


# Training set

## Naming convensions

- 'Image-name-of-BCSS'+'+index'+'[abcd]'.png  -> example: TCGA-A1-A0SK-DX1_xmin45749_ymin25055_MPP-0.2500+0[1101].png

- index: index of the cropped patch in a same BCSS image

- [a: Tumor (TUM), b: Stroma (STR), c: Lymphocytic infiltrate (LYM), d: Necrosis (NEC)]

- Each image (WSI patch) in the training set was cropped from a WSI image at a sliding window with height (224) and width (224).

# Validation and test sets

In validation and test sets, we provide patches with height (224) and width (224) with the semantic segmentation labels of each type of tissue in P mode with the following palette:

palette = [0]*15
palette[0:3] = [255, 0, 0]          # Tumor (TUM)
palette[3:6] = [0,255,0]            # Stroma (STR)
palette[6:9] = [0,0,255]            # Lymphocytic infiltrate (LYM)
palette[9:12] = [153, 0, 255]       # Necrosis (NEC)
palette[12:15] = [255, 255, 255]    # White background or exclude

[1] Amgad, M., Elfandy, H., Hussein, H., Atteya, L. A., Elsebaie, M. A., Abo Elnasr, L. S., ... & Cooper, L. A. (2019). Structured crowdsourcing enables convolutional segmentation of histology images. Bioinformatics, 35(18), 3461-3467.
