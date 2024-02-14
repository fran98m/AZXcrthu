from llama_index import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.embeddings import resolve_embed_model
from llama_index.llns.ollama import Ollama

documents=SimpleDirectoryReader('data')

import logger

#Logger Config

logger = logging.getLogger('Chatbot')
logger.setLevel(logging.INFO)

# Crea un manejador de archivo para escribir mensajes de log en un archivo
file_handler = logging.FileHandler('chatbot_log.txt')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

# Crea un manejador de flujo para escribir mensajes de log en la consola
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

# Añade los manejadores al logger
logger.addHandler(file_handler)
logger.addHandler(stream_handler)


try:
    llm=Ollama(model='dolphin-mixtral',
    callback_manager=CallbackManager([StreamingStdoutCallback()]),
    temperature=0,
    max_tokens=256)
    logger.info(f"Modelo cargado estos son los parametros del modelo: temperature={llm.temperature}, max_tokens={llm.max_tokens}")

except Exception as e:
    logger.error(f"Error {e} al cargar el modelo/error loading model",exec_info=True)

#Para enviar un prompt al modelo simple/To send a prompt to the simple model
#llm("Prompt")

