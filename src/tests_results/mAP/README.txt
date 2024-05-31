# convert_gt_xml.py

Convierte los .xml (en carpeta ground-truth que se encuentra dentro de la carpeta input) a formato .txt (contiene los bounding box)

Se corre desde C:\Users\lilia\Documents\TPP\Código GIT\TPP_ObjectDetection\src\tests_results\mAP

Ejemplo: python convert_gt_xml.py

# Correr el jupyter de test_model

Poner en la celda 2 los path correspondientes


# calculate_map_cartucho.py

Pasos previos: quitar de la carpeta detection-results los resultados de capós sin agujeros (type1).

Calcula el mAP

Se corre desde C:\Users\lilia\Documents\TPP\Código GIT\TPP_ObjectDetection\src\tests_results\mAP

Ejemplo: python calculate_map_cartucho.py

Una vez corrido, crea una carpeta llamada \outputs. Agregar fecha y mover a C:\Users\lilia\Documents\TPP\Código GIT\TPP_ObjectDetection\src\tests_results 

Remember: Copiar resultado de la terminal en un .txt llamado test_results_mAP y guardarlo dentro de la carpeta \output_fecha
          Also agregar a carpeta input la fecha

# main.py

Lo utiliza/ejecuta el script alculate_map_cartucho.py
