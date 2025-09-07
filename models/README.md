# Models

This repository does **not** include large pre-trained models directly in the `models/` folder.
Instead, they are uploaded as **release assets** on GitHub.

## How to use

1. Go to the [Releases](../../releases) section of this repository.
2. Download the model files from the latest release assets:

   * `mask_rcnn_inception_v2_coco_2018_01_28.pbtxt` (configuration)
   * `frozen_inference_graph.pb` (weights)
   * `class.names` (COCO class names)
3. Place them in the local `models/` folder to run the project:

```
models/
├─ mask_rcnn_inception_v2_coco_2018_01_28.pbtxt
├─ frozen_inference_graph.pb
└─ class.names
```

## Notes

* The model files are large, so they are **not tracked by Git**.
* Keeping them as release assets ensures the repository remains lightweight while still reproducible.
