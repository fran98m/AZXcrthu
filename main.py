from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.embeddings import resolve_embed_model
from llama_index.llms.ollama import Ollama

documents=SimpleDirectoryReader('data')

import logging

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
    documents = SimpleDirectoryReader('data').load_data()
    logger.info(f'Sample document: {documents[4]}')
except Exception as e:
    logger.error('Error loading data: {e}', exc_info=True)
finally:
    logger.info('Data loaded successfully')

try:
    Settings.embed_model = resolve_embed_model('local:BAII/bge-small-en-v1.5')
except Exception as e:
    logger.error(f'Error loading embed model: {e}', exc_info=True)
finally:
    logger.info('Embed model loaded successfully')


try:
    Settings.llm=Ollama(model='dolphin-mixtral',request_timeout=40)
except Exception as e:
    logger.error(f'Error loading Ollama model: {e}', exc_info=True)
finally:
    logger.info('Ollama model loaded successfully')


try:
    index=VectorStoreIndex.from_documents(documents,)
except Exception as e:
    logger.error(f'Error loading index: {e}', exc_info=True)
finally:
    logger.info('Index loaded successfully')

try:
    query_engine=index.as_query_engine()
    response=('What thesis does the author proposes and what arguments does he gives?')
    logger.info(response)
except Exception as e:
    logger.error(f'Error loading query engine: {e}', exc_info=True)
finally:
    logger.info(response)