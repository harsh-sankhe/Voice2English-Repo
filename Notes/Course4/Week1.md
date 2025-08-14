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

