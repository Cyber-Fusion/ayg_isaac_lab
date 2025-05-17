from isaaclab.managers import SceneEntityCfg

class WalkTheseWaysParams:
    base_name = "Base"
    hip_name = ".*_Hip"
    shank_names = ".*_Shank"
    thigh_names = ".*_Thigh"
    feet_names = ".*_Foot"
    undesired_contact_names = [".*_Shank", ".*_Thigh"]
    termination_contact_names = ["Base", "Camera", ".*_Hip"]
    height_scanner = SceneEntityCfg("height_scanner")
    