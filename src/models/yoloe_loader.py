from ultralytics import YOLOE

NAMES = [
    "cardboard_box_front", "cardboard_box_diagonal", "cardboard_box_tilted",
    "cardboard_box_heavily_tilted", "cardboard_box_stacked", "cardboard_box_collapsed",
    "cardboard_box_damaged",
    "plastic_container_front", "plastic_container_tilted", "plastic_container_stacked",
    "plastic_container_damaged",
    "metal_case_front", "metal_case_tilted", "metal_case_damaged",
    "wooden_crate_front", "wooden_crate_tilted", "wooden_crate_damaged",
    "stacked_boxes", "leaning_box", "displaced_box", "collapsed_box",
    "wrapped_cargo", "open_cargo", "pallet_wrapped", "pallet_open",
    "other_cargo"
]

def load_yoloe_model(model_path="yoloe-v8l-seg.pt", device="cpu"):
    model = YOLOE(model_path).to(device)
    model.set_classes(NAMES, model.get_text_pe(NAMES))
    return model