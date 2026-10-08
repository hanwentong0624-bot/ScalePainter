# Extending ScalePainter

ScalePainter keeps the paper method in a small set of explicit configuration objects and controllers. An extension should preserve that boundary: define a configuration, build the smallest required controller or adapter, and keep each run's configuration and output versioned beside its own evidence.

## Method variants

The canonical variant names are defined in `scalepainter/config.py`. The five method switches are coverage, reclamation, routed terminal handling, mask-size prompting, and phase alignment.

| Variant | Configuration intent |
|---|---|
| `A0` | All five switches disabled. |
| `A1` | Coverage enabled; reclamation, routed terminal handling, mask-size prompting, and phase alignment disabled. |
| `A2` | Coverage, reclamation, and routed terminal handling enabled; mask-size prompting and phase alignment disabled. |
| `A3` | All switches enabled except phase alignment. |
| `A4` | All switches enabled except mask-size prompting. |
| `A_all` | All five switches enabled. |
| `A_wo1` | Coverage disabled; the remaining switches use their full-method settings. |
| `A_wo2` | Reclamation disabled while the other full-method switches remain enabled. It preserves the full parent's initial-scale route conventions and disables reclamation. |

`A_wo3` is an alias of `A4`, and `A_wo4` is an alias of `A3`. They do not add unique runs. The eight unique configurations are therefore `A0`, `A1`, `A2`, `A3`, `A4`, `A_all`, `A_wo1`, and `A_wo2`.

The configuration builder keeps terminal routing and no-reclamation behavior explicit. In particular, do not treat an alias as a separate method or silently change the `A_wo2` route convention while adding a controller.

## Adding a future baseline

Keep a future baseline behind a separate, narrow adapter. The adapter should accept:

- the visible-only image condition;
- the binary mask;
- the original prompt.

It should return raw RGB output together with metadata such as the adapter name, configuration/version, input geometry, seed, and any model or checkpoint identifier available to the caller. The adapter boundary should not require a heavy plugin framework or import the ScalePainter controller graph. This keeps baseline dependencies and failure modes auditable.

The baseline must receive the visible-only condition. Hidden source RGB must not be made available through a convenience argument, preprocessing cache, debug tensor, or metadata field. Any geometry conversion should be explicit and recorded in the returned metadata.

## Geometry contract

Paper evaluation uses a fixed 1024 x 1024 placement. In `paper` mode, the input must already satisfy that placement and the native safe fill is constructed before inference.

Generic inputs preserve aspect ratio. `scalepainter/geometry.py` first creates a safe filled condition, then resamples it and conservatively resizes the mask: a destination pixel is unknown when its source footprint overlaps any masked source pixel. The resized image is letterboxed into the model canvas with edge padding. This preserves the visible context before resampling and records the native dimensions and placement needed for restoration.

Extensions should use the existing geometry object and restoration path rather than introducing a second resize or mask convention. A new geometry policy belongs in a separate, named mode with its own tests and evidence.

## Evidence and versioning rules

An experiment configuration, output directory, and method version are one unit. Give every new experiment a new versioned location. Never overwrite frozen evidence, frozen manifests, or an existing output directory when changing a controller, geometry rule, checkpoint, seed policy, or evaluator.

Keep the configuration and metadata sufficient to identify the variant, alias resolution, dataset split, geometry mode, prompt, mask, seed, checkpoint, and failure status. Preserve failed outputs and error metadata when they are part of the evaluated population.

The current release acceptance state is **not complete**. GPU parity and metric parity are still pending. A new extension must not be described as accepted, reproducible, or paper-parity complete until those gates are independently run and recorded.
