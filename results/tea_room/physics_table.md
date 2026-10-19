## Simulated (run/tea_room/mujoco_seed/scene.xml)

| body | joint | mass_kg | friction |
|---|---|---|---|
| Door0_Door_leaf | hinge | 128.731 | 0.45 |
| Sink_Storage0_Drawer_Right | slide | 0.702 | 0.45 |
| Sink_Storage0_Door_Right | hinge | 0.324 | 0.45 |
| Sink_Storage0_Door_Left | hinge | 0.527 | 0.45 |
| Storage0_door_r | hinge | 1.418 | 0.45 |
| Storage0_door_l | hinge | 1.544 | 0.45 |
| Storage1_Door_Front | hinge | 2.953 | 0.55 |
| Window0_Sash_Sliding_Right | slide | 4.425 | 0.22 |
| Chair0 | free | 5.106 | 0.55 |
| Chair1 | free | 5.617 | 0.55 |
| Chair2 | free | 5.348 | 0.55 |
| Chair3 | free | 5.617 | 0.55 |
| Dishwasher0 | free | 30.881 | 0.55 |

## Where the numbers came from (run/tea_room/scene_init/scene_stage/room_init/room)

| object | category | mass_kg | mass_source | friction | restitution | materials | note |
|---|---|---|---|---|---|---|---|
| Sink_Storage0 | storage | 36.0 | derived | 0.45, 0.55 | 0.15 | -, Cabinet_Laminate |  |
| Storage0 | storage | 55.0 | derived | 0.45, 0.55 | 0.15 | -, White_Laminate_Door |  |
| Storage1 | storage | 24.075 | authored_density, derived | 0.55 | 0.15 | -, White_Enamel |  |
| Storage3 | storage | 48.0 | derived | 0.45, 0.55 | 0.15 | -, Cream_Laminate |  |
| Table0 | table | 19.0 | derived | 0.55 | 0.15 | - |  |
| Wall10_Door_0 | door | 42.703 | authored_density, derived | 0.45, 0.55 | 0.15 | -, Mahogany_Laminate | total mass 19.21 kg derived from door occupancy density over a 0.213 m^3 bounding box |
| Wall10_Window_0 | window | 36.0 | derived | 0.22, 0.55 | 0.15 | Frame_Brown_uPVC |  |
| ChairCluster0 | chair | 9.177 | derived | 0.55 | 0.15 | - | total mass 9.18 kg derived from chair occupancy density over a 0.367 m^3 bounding box |
| ChairCluster1 | chair | 10.33 | derived | 0.55 | 0.15 | - | total mass 10.33 kg derived from chair occupancy density over a 0.413 m^3 bounding box |
| ChairCluster2 | chair | 11.132 | derived | 0.55 | 0.15 | - | total mass 11.13 kg derived from chair occupancy density over a 0.445 m^3 bounding box |
| Dishwasher0 | dishwasher | 30.456 | derived | 0.55 | 0.15 | - | total mass 30.46 kg derived from dishwasher occupancy density over a 0.254 m^3 bounding box |
| Sofa0 | sofa | 18.935 | derived | 0.55 | 0.15 | - | total mass 18.94 kg derived from sofa occupancy density over a 0.631 m^3 bounding box |

MuJoCo has no restitution parameter: the restitution column is carried for other engines and is not simulated.
