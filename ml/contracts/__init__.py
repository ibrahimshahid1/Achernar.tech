"""Frozen input/output contracts shared by the training and runtime sides.

A contract lives here when two independently-written implementations must agree
on it exactly — typically the Python trainer and the Rust daemon. The FQE learned
this the hard way: three files each described the model's input differently, and
nothing detected the disagreement until `rebuild_fqe` added a parity gate.
"""
