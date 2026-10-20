# Validation and limitations

Check background rejection, neuron masks, overlapping structures, and temporal traces against the original movie and appropriate references. Keep weights, exact source commit, input identity, parameters, and output mappings with every run.

Checkpoint compatibility shims are documented in the root README; loading an older model does not establish scientific equivalence. Validate any changed runtime/model combination on a known recording.

The docs build performs no model execution, training, plugin validation, or manuscript reproduction.
