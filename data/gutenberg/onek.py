import requests
import os
import time

def descargar_1000_libros():
    url_api = "https://gutendex.com/books/"
    carpeta_destino = "silver"
    
    # Crear la carpeta si no existe
    if not os.path.exists(carpeta_destino):
        os.makedirs(carpeta_destino)
    
    libros_descargados = 0
    siguiente_pagina = url_api
    
    print(f"Iniciando descarga en la carpeta: {os.path.abspath(carpeta_destino)}")

    while siguiente_pagina and libros_descargados < 1000:
        try:
            r = requests.get(siguiente_pagina)
            datos = r.json()
        except Exception as e:
            print(f"Error de conexión: {e}")
            break

        for libro in datos['results']:
            if libros_descargados >= 1000:
                break
            
            # Buscar el link del formato TXT
            formatos = libro.get('formats', {})
            url_txt = None
            for mime, url in formatos.items():
                if 'text/plain' in mime and url.endswith('.txt'):
                    url_txt = url
                    break
            
            if url_txt:
                try:
                    # Nombre de archivo limpio
                    titulo_limpio = "".join([c for c in libro['title'] if c.isalnum() or c in (' ', '_')]).rstrip()
                    nombre_archivo = f"{libro['id']}_{titulo_limpio[:50]}.txt"
                    ruta_completa = os.path.join(carpeta_destino, nombre_archivo)
                    
                    # Descargar el contenido del libro
                    print(f"[{libros_descargados + 1}/1000] Descargando: {libro['title']}...")
                    contenido = requests.get(url_txt)
                    
                    with open(ruta_completa, 'w', encoding='utf-8') as f:
                        f.write(contenido.text)
                    
                    libros_descargados += 1
                    
                    # IMPORTANTE: Pausa para no ser bloqueado por Gutenberg
                    time.sleep(2) 
                    
                except Exception as e:
                    print(f"No se pudo descargar el libro {libro['id']}: {e}")
            
        siguiente_pagina = datos.get('next')

    print(f"\n¡Proceso finalizado! Se descargaron {libros_descargados} libros en '{carpeta_destino}'.")

if __name__ == "__main__":
    descargar_1000_libros()