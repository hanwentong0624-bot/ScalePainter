# Third-party sources

The `infinity/` package derives from [FoundationVision/Infinity](https://github.com/FoundationVision/Infinity) at Git commit `9f9fcd1de7ead377ddf1106db1249a35f3a59143`. Its MIT license and FoundationVision attribution are preserved in `third_party/Infinity-LICENSE`.

This distribution selects inference dependencies, removes inactive research hooks and unused training support, preserves the recorded RoPE compatibility change, and exposes ScalePainter's token, residual and accumulation callbacks. The Git source commit and the Hugging Face model revision are different identifiers and must not be interchanged.

Evaluation uses installed torchmetrics, ImageReward, HPSv2, Transformers and their model assets. It preserves the project's recorded FreeInpaint/Token Painter metric conventions while removing the implicit import from a separate FreeInpaint research checkout. Dependency and model licenses remain those of their respective providers.

No Token-Painter source, dataset images, model checkpoint files, historical experiment outputs or internal research logs are included in this package.
