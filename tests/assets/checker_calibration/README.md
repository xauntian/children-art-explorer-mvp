# Checker calibration fixtures

These eight images support the frozen development cases in `tools/calibrate_checker.py` and its offline rule tests. They are engineering fixtures, not an independent acceptance set. Their expected labels do not establish model accuracy or user approval.

- `originals/`: three synthetic marker drawings created for development (`debug_01_sun_house.jpg`, `debug_03_shapes.jpg`, `debug_04_flower.jpg`). They are not the three child-drawing reference images used in visual experiments.
- `generated/`: five retained local-model outputs from those development drawings. Four keep their original descriptive filenames; `house_textured.png` is the retained textured-house candidate.

The calibration tool combines these files with deterministic perturbations to prepare eleven cases. Keep the original image bytes unchanged so hashes and pixel-level checks remain reproducible. Runtime records, machine paths and model weights are not part of these fixtures.
