# Data contracts

SSFA operates on calcium-imaging movies and produces background-rejected movies, segmentation products, and temporal traces. The Hydra orchestration exposes background and segmentation stages separately.

The full pipeline writes `bg_stage/`, `seg_stage/`, and `final_output/` under the selected output directory. The web UI exports TIFF products, a temporal-trace MAT file, a manifest, and a ZIP bundle.

Treat exported traces as calcium-related measurements, not automatically inferred electrophysiology spikes. Before downstream registration/deconvolution, verify the actual MAT fields, component order, footprint orientation, sampling rate, normalization, and any crop/resampling from the producing code.

There is no claim here that the SSFA export is directly interchangeable with another repository's adapter format.
