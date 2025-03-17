import ollama

# Texto de prueba (reemplaza esto con un caso real)
text = "Ejemplo de caso limpio"

# Generar los embeddings
embedding = ollama.embeddings(model="mxbai-embed-large", prompt=text)

# Inspeccionar la estructura del resultado
print(embedding)

