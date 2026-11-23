"""capture — room recordings from other capture apps, turned into LiteReality scans.

    stray   Stray Scanner (free iOS app, Stray Robots): a 60 fps LiDAR video with per-frame ARKit
            poses, depth and confidence -> our scan folder (frame_*.jpg/.json, depth_*, conf_*)
    align   one gravity-aligned ARKit world onto another (yaw about Y + a translation), so a
            recording lines up with the room's LiteReality Scanner scan and its RoomPlan layout
"""
