# TPP_ObjectDetection
Object Detection Code for Final Project

# Entrenar la red
python models/research/object_detection/model_main_tf2.py --pipeline_config_path=models/mymodel/pipeline_file.config --model_dir=checkpoints/ --alsologtostderr --num_train_steps=40000 --sample_1_of_n_eval_examples=1

# Convertir a .tflite (creo)
python models/research/object_detection/export_tflite_graph_tf2.py --trained_checkpoint_dir=TPP_ObjectDetection\src\checkpoints\checkpoints_20240509 --output_directory=TPP_ObjectDetection\src\custom_tflite_models\custom_tflite_model_20240512 --pipeline_config_path=TPP_ObjectDetection\src\pipelines\pipeline_file.config
