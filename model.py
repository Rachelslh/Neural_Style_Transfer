
from PIL import Image
import numpy as np
import tensorflow as tf


class NeuralStyleTransferModel:
    def __init__(self, content_image_path, style_image_path, backbone_model, pretrained_weights, content_layer, style_layers_with_weights, alpha, beta, img_size, learning_rate=0.01, **kwargs) -> None:
        content_image = np.array(Image.open(content_image_path).resize((img_size, img_size)))
        self.content_image = tf.constant(np.reshape(content_image, ((1,) + content_image.shape)))
        
        style_image =  np.array(Image.open(style_image_path).resize((img_size, img_size)))
        self.style_image = tf.constant(np.reshape(style_image, ((1,) + style_image.shape)))        
        
        self.preprocessed_content =  tf.Variable(tf.image.convert_image_dtype(self.content_image, tf.float32))
        self.preprocessed_style =  tf.Variable(tf.image.convert_image_dtype(self.style_image, tf.float32))
        
        self.style_layers_weights = style_layers_with_weights.values()
        
        if backbone_model == "vgg":
            backbone_model = tf.keras.applications.VGG19(include_top=False,
                                            input_shape=(img_size, img_size, 3),
                                            weights=pretrained_weights)

            backbone_model.trainable = False
        else:
            raise NotImplementedError()
            
        layers_outputs = [backbone_model.get_layer(layer).output for layer in [content_layer] + list(style_layers_with_weights.keys())]
        self.feature_model = tf.keras.Model([backbone_model.input], layers_outputs)
        
        self.content_encodings: tf.Tensor = self.feature_model(self.preprocessed_content)[0]
        self.style_encodings: tf.Tensor = self.feature_model(self.preprocessed_style)[1:]
    
        self.alpha = alpha
        self.beta = beta
        self.optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
        
        self.generate_correlated_image_with_content_image()
        
        
    def generate_correlated_image_with_content_image(self):
        self.generated_image = tf.image.convert_image_dtype(self.content_image, tf.float32)
        noise = tf.random.uniform(tf.shape(self.generated_image), -0.25, 0.25)
        self.generated_image = tf.add(self.generated_image, noise)
        self.generated_image = tf.Variable(tf.clip_by_value(self.generated_image, clip_value_min=0.0, clip_value_max=1.0))


    def compute_content_cost(self, generated_encodings: tf.Tensor):
        # Retrieve dimensions from a_G
        m, n_H, n_W, n_C = self.content_encodings.get_shape().as_list()
        
        # Unroll encodings (activations) into (m, n_h * n_w, n_c)
        a_C_unrolled = tf.reshape(self.content_encodings, (m, -1, n_C))
        a_G_unrolled = tf.reshape(generated_encodings, (m, -1, n_C))
        
        cost = tf.reduce_sum(tf.square(a_C_unrolled - a_G_unrolled)) / (4.0 * n_H * n_W * n_C)
        
        return cost


    def compute_layer_style_cost(self, style_encodings: tf.Tensor, generated_encodings: tf.Tensor):
        # Retrieve dimensions from a_G
        _, n_H, n_W, n_C = style_encodings.get_shape().as_list()
        
        # Reshape the tensors from (1, n_H, n_W, n_C) to (n_C, n_H * n_W)
        a_S = tf.transpose(tf.reshape(style_encodings, (-1, n_C)))
        a_G = tf.transpose(tf.reshape(generated_encodings, (-1, n_C)))

        # Computing gram_matrices for both images S and G
        GS = gram_matrix(a_S)
        GG = gram_matrix(a_G)

        # Computing the loss
        layer_cost = tf.reduce_sum(tf.square(GS - GG))/(4.0 *(( n_H * n_W * n_C)**2))
        
        return layer_cost


    def compute_style_cost(self, generated_encodings_per_layer, layers_weights):
        # initialize the overall style cost
        cost = 0

        for (style_encodings, generated_encodings, weight) in zip(self.style_encodings, generated_encodings_per_layer, layers_weights):  
            # Compute style_cost for the current layer
            layer_cost = self.compute_layer_style_cost(style_encodings, generated_encodings)
            # Add weight * J_style_layer of this layer to overall style cost
            cost += weight * layer_cost

        return cost


    def compute_total_cost(self, content_cost, style_cost):
        return self.alpha * content_cost + self.beta * style_cost

    
    @tf.function()
    def on_train_step(self):
        with tf.GradientTape() as tape:
            tape.watch(self.generated_image)
            # Compute a_G as the vgg model output for the current generated image
            generated_encodings_per_layer = self.feature_model(self.generated_image)
            generated_encodings_content_layer = generated_encodings_per_layer[0]
            generated_encodings_style_layers = generated_encodings_per_layer[1:]
            
            # Compute the style cost
            style_cost = self.compute_style_cost(generated_encodings_style_layers, self.style_layers_weights)

            # Compute the content cost
            content_cost = self.compute_content_cost(generated_encodings_content_layer)
            # Compute the total cost
            cost = self.compute_total_cost(content_cost, style_cost)
            
        grad = tape.gradient(cost, self.generated_image)

        self.optimizer.apply_gradients([(grad, self.generated_image)])
        self.generated_image.assign(tf.clip_by_value(self.generated_image, clip_value_min=0.0, clip_value_max=1.0))
        
        
def gram_matrix(A):
    return tf.matmul(A, tf.transpose(A))


def tensor2image(tensor: tf.Tensor):
    tensor = tensor[0] * 255
    tensor = np.array(tensor, dtype=np.uint8)
    return Image.fromarray(tensor)