"""Auto-annotation (stub).

Plan: run the strongest available detector (PyTorch RF-DETR, or an ensemble) over
unlabelled images and emit pre-labels for human review (e.g. via supervision /
CVAT formats), to bootstrap fine-tuning datasets. Grow this when a labelling need
actually appears.
"""
