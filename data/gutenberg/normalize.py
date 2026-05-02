import os
import string

def normalizar_libros():
    carpeta_origen = "silver"
    carpeta_destino = "normalized"
    
    # Crear la carpeta de destino si no existe
    if not os.path.exists(carpeta_destino):
        os.makedirs(carpeta_destino)
        
    archivos = [f for f in os.listdir(carpeta_origen) if f.endswith('.txt')]
    total = len(archivos)
    
    if total == 0:
        print(f"No se encontraron archivos .txt en '{carpeta_origen}'.")
        return
        
    print(f"Iniciando normalización de {total} archivos...")
    
    # Combinar puntuación estándar con caracteres tipográficos comunes en libros
    puntuacion_extra = "“”‘’«»—–…¿¡"
    toda_puntuacion = string.punctuation + puntuacion_extra
    
    # Crear una tabla de traducción altamente eficiente
    tabla_puntuacion = str.maketrans('', '', toda_puntuacion)
    
    for i, archivo in enumerate(archivos, 1):
        ruta_origen = os.path.join(carpeta_origen, archivo)
        ruta_destino = os.path.join(carpeta_destino, archivo)
        
        try:
            with open(ruta_origen, 'r', encoding='utf-8') as f:
                texto = f.read()
                
            # 1. Convertir todo a minúsculas
            texto_normalizado = texto.lower()
            
            # 2. Eliminar toda la puntuación
            texto_normalizado = texto_normalizado.translate(tabla_puntuacion)
            
            # Guardar el archivo normalizado
            with open(ruta_destino, 'w', encoding='utf-8') as f:
                f.write(texto_normalizado)
                
            # Imprimir progreso cada 100 libros
            if i % 100 == 0:
                print(f"Progreso: {i}/{total} libros procesados.")
                
        except Exception as e:
            print(f"Error al procesar el archivo {archivo}: {e}")

    print(f"\n¡Normalización completada! {total} libros guardados en '{carpeta_destino}'.")

if __name__ == "__main__":
    normalizar_libros()