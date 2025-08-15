# COURSE 3
## WEEK 1
- CNNs are neural networks for computer vision tasks like image classification (e.g., cat or not), object detection (e.g., locate cars in a photo), and neural style transfer (e.g., make a photo look like a Picasso painting). Regular neural networks struggle with large images (e.g., 1000x1000x3 = 3M features), needing billions of parameters (e.g., 1,000 x 3M = 3B for one layer), causing overfitting and slow training. CNNs use convolutions—small filters that scan the image to detect patterns like edges—with far fewer parameters. This makes them efficient and effective for big images, powering apps like self-driving cars and face recognition.

## Why Convolutions Matter:

1. CNNs are great for computer vision because they detect patterns in images step-by-step:
 - Early layers find simple features like edges (e.g., lines in a photo).
 - Later layers combine edges to detect parts of objects (e.g., a car’s wheel).
 - Even later layers recognize whole objects (e.g., a car or a face).
 - The convolution operation is how CNNs find these edges efficiently, starting with something like vertical edge detection.


2. What is a Convolution?:
A convolution applies a small matrix called a filter (or kernel) to an image to detect patterns, like vertical edges.
Example: For a 6x6 grayscale image (a matrix of pixel values), you use a 3x3 filter to scan the image and create a new, smaller image (4x4) that highlights edges.
The filter “slides” over the image, checking small regions at a time, making it computationally efficient.


3. How Convolution Works (Vertical Edge Detection Example):

Image: A 6x6 grayscale image (no RGB, just one channel for simplicity).
Filter: A 3x3 matrix, e.g., for vertical edge detection:
$$\begin{bmatrix}
1 & 0 & -1 \\
1 & 0 & -1 \\
1 & 0 & -1
\end{bmatrix}$$

This filter looks for bright pixels on the left and dark pixels on the right (a vertical edge).
Output: A 4x4 matrix (or image) where high values indicate vertical edges.


4. Why It Detects Vertical Edges:

- In a simplified 6x6 image where the left half is bright (pixel value 10) and the right half is dark (0), there’s a clear vertical edge in the middle.
- The filter [1, 0, -1; 1, 0, -1; 1, 0, -1] gives high values (e.g., 30) when it sees bright pixels on the left and dark on the right, highlighting the edge in the output.
- The output image shows a bright strip where the edge is, confirming the filter detected the vertical edge.


5. Why Convolutions Are Efficient:
Unlike regular neural networks, which need billions of parameters for large images (e.g., 1000x1000), convolutions use small filters (e.g., 3x3) that scan the image, reusing the same filter everywhere.
This reduces parameters, making CNNs faster and less prone to overfitting (learning noise from too many parameters).


6. Note:
The convolution operation is denoted by an asterisk (*) in math, but in programming (e.g., Python, TensorFlow), it’s implemented with functions like conv2d or tf.nn.conv2d, not the * symbol, to avoid confusion with multiplication.

## EDGE DETECTION 
Convolutions help CNNs detect edges (e.g., vertical or horizontal lines) in images, a first step in recognizing objects.
Edges are transitions between light and dark pixels, like the outline of a building or road sign.

1. Positive vs. Negative Edges:

- Positive Edge: Light-to-dark transition (e.g., bright on left, dark on right).
   - Example: 6x6 image with 10s (bright) on left, 0s (dark) on right, convolved with a 3x3 vertical edge filter [1,0,-1; 1,0,-1; 1,0,-1].
   - Output: 4x4 matrix with high values (e.g., 30) in the middle, showing a vertical edge.

- Negative Edge: Dark-to-light transition (e.g., dark on left, bright on right).
Same filter gives negative values (e.g., -30) in the output, indicating the reverse transition.


2. Types of Edge Detectors:
- Vertical Edge Filter: [1,0,-1; 1,0,-1; 1,0,-1] detects bright-to-dark vertical transitions.
Horizontal Edge Filter: [1,1,1; 0,0,0; -1,-1,-1] detects bright-to-dark horizontal transitions (bright on top, dark on bottom).
Example: In a 6x6 image with bright top-left and bottom-right corners, the horizontal filter gives high values (e.g., 30) for bright-to-dark horizontal edges and negative values (e.g., -30) for dark-to-bright edges.


3. Other Filters:
- Sobel Filter: [1,2,1; 0,0,0; -1,-2,-1] gives more weight to central pixels, making it more robust for vertical edges.
Scharr Filter: [3,10,3; 0,0,0; -3,-10,-3] has different weights for even better edge detection.
These are alternatives to the basic vertical/horizontal filters.

- Convolutions use small filters (e.g., 3x3) across the entire image, reducing parameters compared to fully connected layers.
This makes CNNs computationally efficient and less prone to overfitting, ideal for large images.

---
## padding 

1. What is Padding?

- Add zero-pixel border around image before convolution to control output size.
   - Example: 6x6 image + 1-pixel padding → 8x8. Convolve with 3x3 filter → 6x6 output.

2. Why Use Padding?
- Prevents Shrinking: Without padding, 6x6 → 4x4 with 3x3 filter. Padding keeps size stable.
- Preserves Edges: Ensures corner/edge pixels are used, avoiding loss of info.
- Crucial for deep networks (e.g., 100 layers) to avoid tiny outputs.


3. Types of Convolution:
- Valid: No padding ($ p=0 $), output: $ (n - f + 1) \times (n - f + 1) $.
- Same: Pad to keep output size = input size ($ p = \frac{f - 1}{2} $), e.g., $ p=1 $ for 3x3, $ p=2 $ for 5x5.
- Formula: Output size = $ (n + 2p - f + 1) \times (n + 2p - f + 1) $.

4. Odd-Sized Filters:
- Common: 3x3, 5x5 (odd $ f $) for symmetric padding and clear central pixel.
   - Example: 3x3 filter needs $ p=1 $ for same convolution.

## Strided convolutions
Strided convolutions move the filter by $ s $ pixels (e.g., stride = 2) instead of 1, reducing output size. For an $ n \times n $ image, $ f \times f $ filter, padding $ p $, stride $ s $, output is $ \left\lfloor \frac{n + 2p - f}{s} + 1 \right\rfloor \times \left\lfloor \frac{n + 2p - f}{s} + 1 \right\rfloor $. Example: 7x7 image, 3x3 filter, $ p = 0 $, $ s = 2 $ → 3x3 output. Deep learning uses cross-correlation (no filter flip) but calls it convolution, simplifying code without affecting CNN performance.

## convolution over volume
- Convolution on 3D images (e.g., 6x6x3 RGB) using 3D filters (e.g., 3x3x3).
    Filter’s channels ($ n_c $) match image’s channels.
   -Example: 6x6x3 image, 3x3x3 filter → 4x4x1 output.
 
1. How It Works:
Slide 3D filter, multiply 27 values (e.g., 3x3x3) with image region, sum for one output.
Example: Filter with [1,0,-1; 1,0,-1; 1,0,-1] in red, zeros in green/blue detects red vertical edges.

2. Multiple Filters:

Use $ k $ filters (e.g., 2 for vertical, horizontal edges) → stack outputs into $ (n - f + 1) \times (n - f + 1) \times k $ (e.g., 4x4x2).
Formula: Output = $ (n - f + 1) \times (n - f + 1) \times k $ (stride = 1, no padding).
