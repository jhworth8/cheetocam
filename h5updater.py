import tensorflow as tf

# Load the older model
old_model_path = 'cat_detector.h5'
new_model_path = 'cat_detector_updated.h5'

model = tf.keras.models.load_model(old_model_path, compile=False)

# Re-save the model in a format compatible with TensorFlow 2.18
model.save(new_model_path)
print("Model successfully updated and saved.")
