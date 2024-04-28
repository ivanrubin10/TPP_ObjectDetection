# Convert exported graph file into TFLite model file
import tensorflow as tf
import os
import sys

def main():

    if len(sys.argv) != 3:
        print("Usage: python script.py <path>")
        return

        # Get the path argument
    pato = sys.argv[1]
    ruta = sys.argv[2]
        
    if os.path.exists(pato) and os.path.isdir(pato):
        converter = tf.lite.TFLiteConverter.from_saved_model(pato)
        tflite_model = converter.convert()

    with open(ruta, 'wb') as f:
        f.write(tflite_model)

main()