// downsample_bicubic_2x(image)	Resize the image to ½ its original size using bicubic interpolation
// downsample_bicubic_4x(image)	Resize the image to ¼ its original size using bicubic interpolation
// downsample_bicubic_8x(image)	Resize the image to ⅛ of its original size (often used in multi-scale ops)
// resize_nearest(image, scale)	Fastest resize, but blocky artifacts
// resize_bilinear(image, scale)	Better than nearest, but can be blurry on edges
// resize_bicubic(image, scale)	Resize using bicubic interpolation (e.g., 0.5 = downsample by 2×)
// resize_bicubic_to(image, width, height)	Resize to an exact resolution with bicubic interpolation
// resize_lanczos(image, scale)	High-quality downsampling for photo realism (slower, uses sinc kernel)
// resize_area(image, scale)	Average-pixel-based method, often used for large downsampling in OpenCV
