MALIGNANT_CLASSES = {"AKIEC", "BCC", "MAL_OTH", "MEL", "SCCKA"}


def make_binary_label(class_name: str) -> str:
    return "malignant" if class_name in MALIGNANT_CLASSES else "benign"

