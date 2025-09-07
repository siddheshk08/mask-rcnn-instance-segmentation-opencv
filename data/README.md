# Data

This repository does **not** include datasets directly. Only small sample images are allowed for quick testing.

## How to use

1. Create a subfolder `data/samples/`.
2. Place your test images inside it, for example:

```
data/
└─ samples/
   └─ cat_and_dog.png
```

3. Use these images to quickly verify that the model and pipeline are working.

## Notes

* Large datasets are **not tracked by Git**.
* For training or extended evaluation, you can download datasets externally (e.g., COCO, Pascal VOC) and place them locally.
* The project is designed to work with **any image inputs** as long as they are placed in `data/samples/`.
