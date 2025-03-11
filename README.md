Dataset utilizado se encuentra en el siguiente link  'https://drive.google.com/file/d/17bFad1mvobn23OFP3eMiqnGTjhirCUZJ/view?usp=drive_link'

---- CODIGOS ----
Codigo1modelo.py: CODIGO PARA ENTRENAR EL MODELO, UTILIZANDO CUDA (SE REQUIERE TARJETA DE VIDEO NVIDIA). PERO SE PUEDE MODIFICAR PARA NO USAR LA GPU Y USAR EL CPU.
Codigo2robot.py: CODIGO PARA EJECUTAR EL MODELO EN EL ROBOT PICAR-X.
prueba.py: CODIGO PARA EJECUTAR EL MODELO ENTRENADO .H5 *SE NECESITA PYTHON VERSION 3.9 NUMPY, CV2 y TENSORFLOW VERSION 2.10 para el codigo de ejecución del modelo .h5*

Para poder ejecutar códigos
pip install numpy pandas tensorflow opencv-python pillow matplotlib scikit-learn 

*PARA EJECUTAR CORRECTAMENTE EL MODELO EN EL ROBOT PICAR-X, SE DEBE CONVERTIR EL MODELO .H5 A .TFLITE*

