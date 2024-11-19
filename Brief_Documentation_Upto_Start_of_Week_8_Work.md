# 🌊 Iceberg Detection Project: A Deep Learning Approach
> Satellite Radar Image Classification for Maritime Safety

## 📊 Project Statistics
- **Final Accuracy**: 89.41%
- **Dataset Size**: 1,604 training images, 8,424 test images
- **Model Type**: Custom Convolutional Neural Network
- **Training Environment**: Google Colab

## 🗓️ Weekly Development Progress

### Week 1-2: Data Preprocessing & Setup
```python
train = pd.read_json('/content/extracted_files/data/processed/train.json')
test = pd.read_json('/content/extracted_files/data/processed/test.json')

def scale_images(data_frame):
    image_list = []
    for index, data_row in data_frame.iterrows():
        band1 = np.array(data_row['band_1']).reshape(75, 75)
        band2 = np.array(data_row['band_2']).reshape(75, 75)
        combined_band = band1 + band2
        
        normalized_band1 = (band1 - band1.mean()) / (band1.max() - band1.min())
        normalized_band2 = (band2 - band2.mean()) / (band2.max() - band2.min())
        normalized_combined = (combined_band - combined_band.mean()) / 
                            (combined_band.max() - combined_band.min())
        
        image_list.append(np.dstack((normalized_band1, 
                                   normalized_band2, 
                                   normalized_combined)))
    return np.array(image_list)
```

#### Key Features
- Data extraction from compressed files
- Three-channel image creation from dual-band data
- Normalization of radar signals
- Data visualization implementation

### Week 2-4: CNN Implementation
```python
neural_net = Sequential([
    Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(75, 75, 3)),
    MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
    Dropout(0.2),
    
    Conv2D(128, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Dropout(0.2),
    
    Conv2D(128, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Dropout(0.2),
    
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Dropout(0.2),
    
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.2),
    Dense(256, activation='relu'),
    Dropout(0.2),
    Dense(1, activation="sigmoid")
])
```

### Week 4: Model Training Setup
```python
opt = Adam(learning_rate=1e-4)
neural_net.compile(loss='binary_crossentropy', 
                  optimizer=opt, 
                  metrics=['accuracy'])

callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath="best_model_weights.keras",
        monitor="val_accuracy",
        save_best_only=True,
        mode="max"
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_accuracy",
        factor=0.9,
        patience=2,
        min_lr=1e-6,
        mode="max",
        verbose=True
    ),
    keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        mode="max",
        verbose=True
    )
]
```

### Week 5: Initial Training
```python
training_history = neural_net.fit(
    X_train, Y_train,
    batch_size=10,
    epochs=100,
    verbose=1,
    validation_split=0.2,
    callbacks=callback_functions
)
```

#### Training Progress
| Epoch | Training Accuracy | Validation Accuracy |
|-------|------------------|-------------------|
| 1     | 50.42%          | 56.81%           |
| 5     | 76.66%          | 82.10%           |
| 10    | 84.72%          | 88.72%           |
| 15    | 88.03%          | 88.72%           |

### Weeks 6-8: Optimization & Testing

#### Learning Rate Evolution
```python
Initial: 1e-4    # Week 6
→ 8.99e-5       # Week 7
→ 8.10e-5       # Week 7
→ 6.56e-5       # Week 8
→ 5.31e-5       # Week 8 Final
```

#### Weekly Progress
- Week 6: 52.81% accuracy
- Week 7: 76.36% → 84.16%
- Week 8: 87.20% → 91.50%
- Final Test: 89.41%

#### Image Classification Implementation
```python
def classify_image(index, data, labels, model, threshold=0.5):
    image = np.expand_dims(data[index], axis=0)
    probability_of_iceberg = model.predict(image)[0][0]
    
    predicted_label = "Iceberg" if probability_of_iceberg >= threshold else "Ship"
    actual_label = "Iceberg" if labels[index] == 1 else "Ship"
    
    plt.imshow(data[index])
    plt.title(f"Prediction: {predicted_label}, Actual: {actual_label}\n"
              f"Iceberg Probability: {probability_of_iceberg:.2f}")
    plt.axis('off')
    plt.show()
```

## 🎯 Future Development
### Week 9-10
1. **UI Development**
   - Interface design
   - Real-time prediction visualization
   - User feedback system

2. **Maritime Integration**
   - System compatibility testing
   - Performance optimization
   - Deployment preparation

## 🛠️ Dependencies
```bash
tensorflow
keras
pandas
numpy
matplotlib
seaborn
py7zr
```

## 👥 Team
- MD JAWAD KHAN (202381977)
- SYED MUDASSIR HUSSAIN (202387913)

---
> 📝 This project aims to enhance maritime safety through advanced machine learning techniques.
