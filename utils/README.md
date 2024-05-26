Estos son los 3 scritps que nos sirven para preparar la información para el entrenamiento. Se ejecutan en el siguiente orden: 

Paso previo, haber etiquetado las imágenes utilizando el programa labelimg.

# split.py

Separa las imagenes (y sus archivos .xml) en grupo de train validation y test. se le tiene que indicar el porcentaje que se quiere asignar a train y validation. El resto se designa a test. 

    # modo de uso:

    split.py --images={input_path} --train={(0..1)} --val={(0..1)}

    # ejemplo:

    python utils/split.py --images=data/labelled_images/images --train=0.75 --val=0.15

    # salida: 

    3 carpetas con imágenes llamadas train test validation dentro del directorio de una jerarquia superior al input_path. (En este caso "images").

# create_csv.py:

Crea a partir de los archivos .xml de cada foto, una lista en .csv con la información resumida en un solo archivo
Los crea en la carpeta data/labelled_images. Luego moverlos a data/etc

    # modo de uso:

    create_csv.py {input_path}

    # ejemplo: 

    python utils/create_csv.py data/labelled_images/train
    python utils/create_csv.py data/labelled_images/validation

    #salida: 

    Archivo .csv llamado train_labels (en este caso train, si el path era images\validation sería validation_labels) dentro de la carpeta "images". 
    Obs.: Chequear que el archivo creado tenga peso, sino hubo algún error.

# create_tfrecord.py:

A partir del csv, el labelmap (.txt con etiquetas en /data/etc) y las imagenes; crea un archivo .tfrecord y otro archivo labelmap.pbtxt. Estos formatos son los que la red puede tomar como input.

    # modo de uso: 

    python create_tfrecord.py --csv_input={input_path} --labelmap={input_path} --image_dir={input_path} --output_path={output_path}

    # ejemplo: 

    python utils/create_tfrecord.py --csv_input=data/etc/train_labels.csv --labelmap=data/etc/labelmap.txt --image_dir=data/labelled_images/train --output_path=data/train.tfrecord

    python utils/create_tfrecord.py --csv_input=data/etc/validation_labels.csv --labelmap=data/etc/labelmap.txt --image_dir=data/labelled_images/validation --output_path=data/validation.tfrecord
    

#  create_tflite.py

A partir del saved_model (checkpoint transformado a .tflite) se crea en el output_path el .tflite para detectar

    # modo de uso: 

    python create_tflite.py --saved_model_path={input_path} --output_path={output_path}

    # ejemplo:

    
    