# Land Use and Land Cover Classification in Western Ghats using Deep Learning

## Overview
This project implements multiclass semantic segmentation for Land Use/Land Cover (LULC) classification in the Western Ghats region of India, specifically focusing on the Idukki area. Using DeepLabV3+ architecture with EfficientNet-B0 backbone, the model classifies satellite imagery into four categories:

1. Settlements
2. Forests
3. Water bodies
4. Vegetated area

![Sample Classification](https://github.com/user-attachments/assets/b577c18f-de6f-4924-983c-1b284b3aae09)

## Dataset
- **Source**: High-resolution satellite images obtained from Google satellites through QGIS
- **Size**: 1564 pre-processed images and corresponding masks
- **Resolution**: 300 dpi
- **Image Format**: RGB (3 spectral bands)
- **Split**: 
  - Training: 1173 images
  - Validation: 391 images

## Model Architecture
The project uses DeepLabV3+ architecture with EfficientNet-B0 backbone for its ability to:
- Capture multi-scale contextual information using atrous convolutions
- Handle complex scene understanding
- Maintain high performance across varying spatial scales

![DeepLabv3+ Architecture](https://github.com/user-attachments/assets/ac0a3ba7-2257-496c-88b0-16b6a952e9dd)

## Performance Metrics
### Training Set
| Metrics    | Vegetated Area | Forest Cover | Settlements | Water Bodies |
|------------|---------------|--------------|-------------|--------------|
| Accuracy   | 0.64          | 0.91         | 0.74        | 0.87         |
| Precision  | 0.73          | 0.83         | 0.77        | 0.92         |
| Recall     | 0.64          | 0.91         | 0.75        | 0.87         |
| F1-Score   | 0.68          | 0.87         | 0.76        | 0.89         |

### Validation Set
| Metrics    | Vegetated Area | Forest Cover | Settlements | Water Bodies |
|------------|---------------|--------------|-------------|--------------|
| Accuracy   | 0.54          | 0.84         | 0.71        | 0.90         |
| Precision  | 0.62          | 0.78         | 0.70        | 0.89         |
| Recall     | 0.54          | 0.84         | 0.71        | 0.90         |
| F1-Score   | 0.57          | 0.81         | 0.70        | 0.90         |

![Model Performance](https://github.com/user-attachments/assets/e2339ed8-d046-4274-a025-4f1ec0e8cb27)


## Setup and Installation

### Prerequisites
- Python >= 3.6
- QGIS (for data collection)

### Dependencies
Install the required packages using pip:

```bash
pip install tensorflow opencv-python numpy matplotlib patchify tifffile pillow splitfolders
```

## Usage

### 1. Data Preparation
Run the data preparation script:

```python
python Data_Preparation.py
```

This script performs the following tasks:
- Converts JSON annotations to image masks
- Creates patches of size 256x256 from the images and masks
- Removes patches with no useful information
- Splits the data into training and validation sets

### 2. Model Training
The model training is implemented in a Jupyter notebook. To train the model:

1. Ensure you have Jupyter Notebook or JupyterLab installed:
   ```bash
   pip install jupyter
   ```

2. Navigate to the project directory and start Jupyter:
   ```bash
   jupyter notebook
   ```

3. Open the `Model Training.ipynb` notebook in the Jupyter interface.

4. Run the cells in the notebook sequentially. The notebook includes:
   - Data loading and preprocessing
   - Model definition (DeepLabV3+ with EfficientNet-B0 backbone)
   - Training configuration
   - Model training
   - Visualization of results
   - Evaluation metrics calculation

Note: Make sure your data paths in the notebook match your local directory structure.

You can adjust these parameters in the notebook as needed.

## Training Configuration
- Batch Size: 8
- Epochs: 25
- Learning Rate: 0.0001
- Optimizer: Adam
- Loss Function: Combined Dice loss and Focal loss
- Activation Functions:
  - Intermediate Layers: ReLU
  - Output Layer: SoftMax
  - 
![Results on testing set](https://github.com/user-attachments/assets/9643e120-da79-4e7e-be3f-2a091150c420)

## Applications
1. **Environmental Monitoring**
   - Forest cover change detection
   - Urban expansion analysis
   - Water body monitoring

2. **Urban Planning**
   - Sustainable development
   - Resource management
   - Infrastructure planning

3. **Education**
   - Integration with geography curriculum
   - Environmental science research
   - Remote sensing studies

## Limitations
- Manual image collection and mask creation is time-consuming
- Requires significant computational resources
- Geographic concentration may limit generalizability
- Temporal bias in image collection

## Future Scope
- Expansion to broader geographical areas
- Integration of temporal analysis
- Addition of more land cover classes
- Implementation of different deep learning architectures
- Integration with other data sources


## Contact
For any queries or suggestions, please open an issue in the repository or contact:
- Email: [purvichoure2@gmail.com](mailto:purvichoure2@gmail.com)
- LinkedIn: [https://www.linkedin.com/in/purvi29/](https://www.linkedin.com/in/purvi29/)

## Citation
If you use this code in your research, please cite:

```bibtex
@inproceedings{
  author    = {Choure, P. and Prajapat, S.},
  title     = {Exploring U-Net, FCN, SegNet, PSPNet, Mask R-CNN and using DeepLabV3+ for Multiclass Semantic Segmentation on Satellite images of Western Ghats},
  booktitle = {The International Conference on Computing, Communication, Cybersecurity \& AI},
  year      = {2024}
}
```
