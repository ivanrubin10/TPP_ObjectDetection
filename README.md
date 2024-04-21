# TPP_ObjectDetection
Object Detection Code for Final Project

# Entrenar la red

python models/research/object_detection/model_main_tf2.py --pipeline_config_path=models/mymodel/pipeline_file.config --model_dir=training/ --alsologtostderr --num_train_steps=40000 --sample_1_of_n_eval_examples=1

# Convertir a .tflite (creo)

<<<<<<< HEAD
python models/research/object_detection/export_tflite_graph_tf2.py --trained_checkpoint_dir {last_model_path} --output_directory {output_directory} --pipeline_config_path {pipeline_file}
=======
python models/research/object_detection/export_tflite_graph_tf2.py --trained_checkpoint_dir {last_model_path} --output_directory {output_directory} --pipeline_config_path {pipeline_file}
>>>>>>> 89751cb564addc1b93163cc6961dccda547f7052
