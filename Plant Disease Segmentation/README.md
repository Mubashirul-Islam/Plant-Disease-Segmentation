This project implements plant leaf disease segmentation using deep learning, specifically a U-Net architecture enhanced with Bi-directional Feature Pyramid Network (BiFPN). The models are trained and evaluated on a dataset of plant leaf images and corresponding segmentation masks.

## Project Structure

- `428-bifpn-segment.ipynb`: Notebook for U-Net with BiFPN model definition, training, and evaluation.
- `unet_428_code.ipynb`: Baseline U-Net model training and evaluation.
- `428-project-compare.ipynb`: Comparison and visualization of predictions from both models.
- `unet_model.h5`: Saved weights for the baseline U-Net model.
- `unet_with_BiFPN.h5`: Saved weights for the U-Net with BiFPN model.
- `segmentation project slides.pptx`: Project presentation slides.

## Model Architectures

- **U-Net**: A standard encoder-decoder architecture for image segmentation.
- **U-Net with BiFPN**: Extends U-Net by integrating BiFPN for improved multi-scale feature fusion, enhancing segmentation accuracy.

## Key Features

- **Data Loading**: Images and masks are loaded and preprocessed to size 128x128.
- **Training**: Both models are trained with Adam optimizer, early stopping, and learning rate reduction callbacks.
- **Evaluation Metrics**: Mean IoU, Dice coefficient, and pixel accuracy are used for performance evaluation.
- **Visualization**: Side-by-side comparison of segmentation results on unseen images.

## Usage

1. **Requirements**
   - Python 3.x
   - TensorFlow
   - Keras
   - NumPy
   - OpenCV
   - scikit-learn
   - Matplotlib

2. **Training**
   - Run unet_428_code.ipynb to train and save the baseline U-Net model.
   - Run 428-bifpn-segment.ipynb to train and save the U-Net with BiFPN model.

3. **Comparison & Visualization**
   - Use 428-project-compare.ipynb to load both models and visualize predictions on unseen images.

## Example: Visualizing Model Predictions

The comparison notebook loads both models and visualizes their predictions:

```python
mod_u = load_model('unet_model.h5')
mod_b = load_model('unet_with_BiFPN.h5')

image, pred_unet = predict_unseen_image(mod_u, image_path, 128, 128)
_, pred_bifpn = predict_unseen_image(mod_b, image_path, 128, 128)

visualize_comparison(image, pred_unet, pred_bifpn)
```

## Results

- The U-Net with BiFPN generally achieves higher segmentation accuracy and better generalization on unseen images compared to the baseline U-Net.

## References

- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- [EfficientDet: Scalable and Efficient Object Detection](https://arxiv.org/abs/1911.09070) (BiFPN)

---

For more details, see the code and results in the provided notebooks.