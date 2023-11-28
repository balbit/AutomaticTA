from llama_index import ServiceContext, VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings import OpenAIEmbedding

embed_model = OpenAIEmbedding()
service_context = ServiceContext.from_defaults(embed_model=embed_model)

# optionally set a global service context to avoid passing it into other objects every time
from llama_index import set_global_service_context

set_global_service_context(service_context)


documents = SimpleDirectoryReader("./data/public").load_data()
print(documents)
index = VectorStoreIndex.from_documents(documents, show_progress=True)

print("Done storing vectors!!")


query_engine = index.as_query_engine()
response = query_engine.query("Which lecture mentions marijuana as an example in one of the problems?")
print(response)