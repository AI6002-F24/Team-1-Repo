# Iceberg Detection Using CNNs (BRIEF README)

An AI-powered system for detecting icebergs in satellite radar images using Convolutional Neural Networks. This project achieves 89.41% accuracy in distinguishing between icebergs and ships in satellite radar imagery (given the limitations of time and resources).
BUT massive improvements can be had.
## Features

- Automated classification of radar images (iceberg vs. ship)
- Real-time processing with <1 second inference time
- Interactive web interface using Streamlit
- Support for both training and testing datasets
- Confidence scores for predictions

## Requirements

- Python 3.10 or later
- Dependencies listed in `requirements.txt`
- Minimum 16GB RAM
- Compatible with Windows 10,11

## Installation

1. Clone the repository:
```bash
git clone https://github.com/AI6002-F24/Team-1-Repo.git
cd Team-1-Repo/final_organized_for_report/iceberg_detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Web Interface

Start the Streamlit application:
```bash
streamlit run app.py
```

### Using the Application

1. **Training Data Analysis**
   - Upload `train.json` (can be found in main parent folder under datasets subfolder)
   - View predictions with actual labels
   - Analyze model performance metrics

2. **New Data Prediction**
   - Upload `test.json` (can be found in main parent folder under dataset subfolder)
   - Get predictions with confidence scores
   - View visualization results

### Data Format Requirements

Input data should be in JSON format with the following structure:
- `train.json`: Contains labeled data with 'band_1', 'band_2', and 'is_iceberg' fields
- `test.json`: Contains unlabeled data with 'band_1' and 'band_2' fields

## Model Architecture

Custom CNN architecture:
- Input Layer: 75×75×3
- 4 Convolutional blocks with MaxPooling and Dropout
- Dense layers (512→256→1)
- Binary classification output

## Performance

- Test Accuracy: 89.41%
- Precision: 0.91
- Recall: 0.84
- F1 Score: 0.87

## Known Limitations

- Limited to binary classification (iceberg vs. ship)
- Dataset size constraint (1,604 training images)
- Resource optimization for deployment environments required

## Authors

- MD JAWAD KHAN (202381977)
- SYED MUDASSIR HUSSAIN (202387913)

## Acknowledgments

- Dataset provided by Statoil/C-CORE Iceberg Classifier Challenge
- Special thanks to Professor and TA for helping all thorugh out the journey

## Repository Structure

```
iceberg_detection/
├── app.py              # Streamlit application
├── model_training.ipynb # Training notebook
├── requirements.txt    # Dependencies
└── README.md          # Setup & usage instructions
```
