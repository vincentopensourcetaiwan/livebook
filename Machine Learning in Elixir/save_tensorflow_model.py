import tensorflow as tf

model = tf.keras.applications.ResNet50(weights="imagenet")

# Create a new model with explicit input layer
inputs = tf.keras.Input(shape=(224, 224, 3), batch_size=None, name='input_1')
outputs = model(inputs)
new_model = tf.keras.Model(inputs=inputs, outputs=outputs)

# Create serving signature
@tf.function(input_signature=[tf.TensorSpec(shape=[None, 224, 224, 3], dtype=tf.float32, name="input_1")])
def serving_function(input_1):
    return {"output_0": new_model(input_1)}

# Save the model with explicit signatures
tf.saved_model.save(
    new_model,
    "resnet",
    signatures={
        "serving_default": serving_function
    }
)
