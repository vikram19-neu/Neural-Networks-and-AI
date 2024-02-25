import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

input_shape = (224, 224, 3) 
num_classes = 3  
l2_lambda = 0.001  
dropout_rate = 0.5  

model = models.Sequential([
   
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape,
                  kernel_regularizer=regularizers.l2(l2_lambda)),
    layers.MaxPooling2D((2, 2)),

    
    layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(l2_lambda)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(l2_lambda)),

    
    layers.Flatten(),

   
    layers.Dense(128, activation='relu'),
    layers.Dropout(dropout_rate),

    layers.Dense(num_classes, activation='softmax')
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


model.summary()
