# Computer Vision and CNNs

- **Computer Vision** is a field of AI that enables machines to interpret and understand visual information from images or videos.

- Common Tasks in Computer Vision are **Image Classification, Object Detection, Face Recognition, Neural Style Transfer**  

---

## CNN for Computer Vision
- **Convolutional Neural Networks (CNNs)** use convolution layers to detect patterns like edges, textures, and shapes.  
- They reduce parameters compared to fully connected networks, making training feasible for large images.  
- CNNs capture spatial relationships in images, enabling accurate and efficient vision models.  

---

## Convolution in CNNs
Convolution allows CNNs to detect patterns such as edges by applying small filters (kernels) across an image.

- Early CNN layers detect simple features (edges, lines).  
- Deeper layers combine these into complex shapes and objects.  

## Example

A grayscale image can be represented as a 2D matrix of pixel brightness values.  
To detect edges, we apply a filter such as:

**Vertical edge detector:**
1 0 -1
1 0 -1
1 0 -1

**Horizontal edge detector:**
1 1 1
0 0 0
-1 -1 -1

### Convolution Process:
1. Place the filter over a 3×3 region of the image.  
2. Multiply each filter value with the corresponding pixel (element-wise multiplication).  
3. Add the results to get a single output value.  
4. Slide the filter across the image and repeat.  

**Output:**  
A 6×6 image with a 3×3 filter produces a 4×4 output matrix representing detected edges, with positive and negative values indicating different edge directions and strengths.

- Other 3 by 3 filters include **Sobel Filter** and **Scharr Filter**

Instead of manually designing filters, **CNNs** can learn the filter values from data using **backpropagation**.  
This allows the network to:  
- Detect edges in any orientation  
- Identify more complex patterns  
- Avoid the need for manual tuning

## Problem with Convolution:
•   Convolution reduces image size.
**Output size = (n − F + 1) × (n − F + 1)**
•	Repeated size reduction in deep networks causes images to shrink too much.
•	Edge/corner pixels are used less often, losing important information.
- Solution for this problems is Padding technique.

## Padding
•	Add a border of extra pixels (usually zeros) around the image before convolution.
•	Preserves more edge information and can keep the same image size after convolution.

Formula with padding:
•	Output size = (n + 2P − F + 1) × (n + 2P − F + 1)
- Where:
    n = input size
	F = filter size
	P = padding size
Example:
•	6×6 image, P = 1, F = 3 → output size = 6×6 (same as input).

### Two common padding types:
1.	**Valid convolution** → P = 0, no padding. Output shrinks: (n − F + 1).
2.	**Same convolution** → choose P so output size = input size.
o	Formula: P = (F − 1) / 2 (works for odd F).
o	Example: F = 3 → P = 1; F = 5 → P = 2.

Why filters are usually odd-sized?
•	Odd sizes have a center pixel (e.g., 3×3, 5×5).
•	Symmetrical padding is easier (no uneven borders).
•	Common in computer vision: 3×3, 5×5, 7×7, and sometimes 1×1 filters.

## Strided Convolutions in CNNs

### What is stride?
•	Stride (s) controls how far the filter moves after each convolution step.
•	Normal convolution: *stride = 1* → filter moves 1 pixel at a time.
•	Strided convolution: *stride > 1* → filter “jumps” more pixels, reducing output size.

Example:
•	Input: 7×7 image
•	Filter: 3×3
•	Padding: P = 0
•	Stride: s = 2
Process:
1.	Place filter and perform element-wise multiplication and then sum.
2.	Move filter 2 pixels right instead of 1.
3.	Repeat across rows and columns.
Output: 3×3 feature map.
Output size formula: **[(N+2P-F)/S]+1 x [(N+2P-F)/S]+1 **
Where:
•	n = input size
•	F = filter size
•	P = padding
•	s = stride
•	Floor ⌊ ⌋ = round down to nearest integer if not whole.

#### Why stride is used ?
- Reduces spatial dimensions (downsampling).
- Decreases computation and parameters.
- Can replace pooling in some architectures.

## Convolution Over 3D Volumes in CNNs
•	An RGB image has three color channels: Red, Green, and Blue.
•	For RGB, we must convolve with 3D filters that match the number of channels.

**How it works**
1.	*Filter placement*: Place the 3×3×3 filter at the top-left of the image volume.
2.	*Element-wise multiply*: Multiply each of the 27 filter values with the corresponding 27 pixels (across R, G, B channels).
3.	*Sum*: Add all products → one output value.
4.	*Slide*: Move filter 1 pixel right, repeat.
5.	Continue row-by-row until the whole image is covered.

Multiple Filters:
•	We can use multiple filters to **detect different features** (e.g., edges, textures).
•	Each filter produces a different output.
•	These feature maps can be stacked, forming a new volume with more channels.


# One-Layer Convolutional Network
