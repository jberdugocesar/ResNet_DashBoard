import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
import numpy as np
import matplotlib as mpl
from torchvision import models, transforms
import torch
import torch.nn as nn
import gdown
from joblib import Memory

# Crear una instancia de caché con el directorio de almacenamiento en memoria
memory = Memory("./cache", verbose=0)

@memory.cache  # Almacenar en caché el modelo
def cargar_modelo():
    # URL de descarga directa del archivo en Google Drive
    file_url = "https://drive.google.com/uc?export=download&id=1qz3DCKhmutALRMndTwboOJiIWpne1G0V"
    gdown.download(file_url, "resnet_model_pytorch.pth", quiet=False)
   
    model  = models.resnet101(weights="DEFAULT")
    num_clases = 6
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_clases)

    ruta_modelo = 'resnet_model_pytorch.pth'  # Ruta del archivo del modelo previamente guardado
    model.load_state_dict(torch.load(ruta_modelo,map_location=torch.device('cpu')))
    model.eval()
    return model

def cargar_modelo_mejorado():
    file_url = "https://drive.google.com/uc?export=download&id=15iQv7NqJS7kq5GstKeyGAt5el4QqXGS7"
    gdown.download(file_url, "new_dataset_pytorch.pth", quiet=False)
    model  = models.resnet101(weights="DEFAULT")
    num_clases = 6
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_clases)

    ruta_modelo = 'resnet_model_pytorch.pth'  # Ruta del archivo del modelo previamente guardado
    model.load_state_dict(torch.load(ruta_modelo,map_location=torch.device('cpu')))
    model.eval()
    return model
    
    
    
    
# Cargar el modelo
model = cargar_modelo()
class_names = ['Cardboard', 'Glass', 'Metal', 'Paper', 'Plastic', 'Trash']

modelV2 = cargar_modelo_mejorado()
classes = ["basura","metal","papel","plastico"]

def main():
    st.title("Aplicación de predicción de imágenes")
    imagen_subida = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])
    if imagen_subida is not None:
        imagen = Image.open(imagen_subida)

         # Verificar si se necesita agregar una dimensión adicional
        if imagen.mode != "RGB":
            imagen = imagen.convert("RGB")

        imagen_np = np.array(imagen)
        imagen_tensor = torch.Tensor(imagen_np).unsqueeze(0)
        st.write("Imagen cargada con éxito y transformada.")

        st.image(imagen, caption='Imagen subida', use_column_width=True)
        prediccion = predecir_imagen(imagen, model)
        clase_predicha = class_names[prediccion]
        
        prediccionv2 = predecir_imagen(imagen,modelV2)
        clase_predicha = classes[prediccionv2]
        
        st.markdown(f"<p style='font-size: 24px;'><span style='color: white;'>La clase predicha por el modelo viejo es: </span><span style='color: red;'>{clase_predicha}</span></p>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size: 24px;'><span style='color: white;'>La clase predicha por el modelo V2 es: </span><span style='color: green;'>{clase_predicha}</span></p>", unsafe_allow_html=True)

def predecir_imagen(imagen, modelo):
    transformaciones = transforms.Compose([
      transforms.Resize((224, 224)),  # Asegura que la imagen tenga el tamaño esperado
      transforms.ToTensor(),  # Convierte la imagen en un tensor
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normaliza los valores de los canales RGB
    ])

    imagen = transformaciones(imagen).unsqueeze(0)  # Preprocesa la imagen
    # Mover la imagen a la GPU si está disponible
    imagen = imagen.to(torch.device('cpu'))

    # Realizar la predicción
    with torch.no_grad():
        salida = model(imagen)
        _, indice = torch.max(salida, 1)
        class_index = indice.item()
    return class_index

def circle_graph():

  # Habilitar el tema de la interfaz de usuario adaptativa
  mpl.style.use('default')
  # Datos para el gráfico de pastel
  labels = ["Papel",
  "Vidrio",
  "Metal",
  "Plástico",
  "Cartón",
  "Basura"]
  values = [594 , 501 , 410 , 482 , 403 , 137]

  # Crear el gráfico de pastel
  fig, ax = plt.subplots()

  # Agregar valores y porcentajes como anotaciones en cada sector
  wedges, texts, autotexts = ax.pie(values, labels=labels, autopct=lambda pct: f'{pct:.1f}%\n({int(pct * sum(values) / 100)})', startangle=90)

  # Configurar aspectos visuales del gráfico
  ax.axis('equal') 
  
  ax.set_title("Cantidad de imagenes por clase")

  plt.setp(autotexts, size=10, weight='bold')

  # Mostrar el gráfico en Streamlit
  st.pyplot(fig)

if __name__ == '__main__':
    # Coloca el código que usa `signal` aquí
    # ...

    # Llama a la función principal de Streamlit
    main()
    st.markdown("---")
    circle_graph()
