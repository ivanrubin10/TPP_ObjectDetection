# TPP_ObjectDetection
Object Detection Code for Final Project

# Entrenar la red

python models/research/object_detection/model_main_tf2.py --pipeline_config_path=models/mymodel/pipeline_file.config --model_dir=training/ --alsologtostderr --num_train_steps=40000 --sample_1_of_n_eval_examples=1

# Convertir a .tflite (creo)

python models/research/object_detection/export_tflite_graph_tf2.py --trained_checkpoint_dir=training/ --output_directory=custom_model_lite/ --pipeline_config_path=models/mymodel/pipeline_file.config
