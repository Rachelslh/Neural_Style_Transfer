import tensorflow as tf
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

from model import NeuralStyleTransferModel, tensor2image


if __name__=="__main__":
    tf.random.set_seed(272)
    
    # Read from config
    config = OmegaConf.load("config.yaml")
    nst_model = NeuralStyleTransferModel(**config)
    
    for i in range(config["epochs"]):
        nst_model.on_train_step()
        print(f"Epoch {i}")
    image = tensor2image(nst_model.generated_image)
    imshow(image)
    image.save(f"images/output_{i}.jpg")
    plt.show() 