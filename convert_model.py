import tensorflow as tf

# The export path contains the name and the version of the model
tf.keras.backend.set_learning_phase(0)
model = tf.keras.models.load_model('./model.h5',compile=False)
export_path = 'dist_model/6'

# Fetch the Keras session and save the model
# The signature definition is defined by the input and output tensors
# And stored with the default serving key
print(model.input)
tf.saved_model.save(model,export_path)