# Import necessary modules
from llama_index import ServiceContext, VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings import OpenAIEmbedding
from llama_index import StorageContext, load_index_from_storage, set_global_service_context
from llama_index.indices.list import SummaryIndexLLMRetriever
from tqdm import tqdm
import shutil
import os


# Set the default contexts and options
embed_model = OpenAIEmbedding()
service_context = ServiceContext.from_defaults(embed_model=embed_model)

# Set the global service context to the created service_context
set_global_service_context(service_context)


def index_from_dir(persist_dir):
    """
    Loads index from persist_dir as the persisted index
    """
    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
    index = load_index_from_storage(storage_context)
    return index


def add_to_index(dir_from, dir_to, persist_dir, was_persisted=True):
    """
    Loads index from persist_dir as the persisted index, 
    adds all documents from dir_from to it, and persists it 
    to persist_dir.

    Then moves all files from dir_from to dir_to
    """
    if not os.listdir(dir_from):
        print("Folder is empty.")
        return

    if was_persisted:
        # Load the index from storage
        index = index_from_dir(persist_dir)

        # Load documents from dir_from
        new_documents = SimpleDirectoryReader(dir_from).load_data()

        # Add new documents to the index
        for doc in tqdm(new_documents):
            index.insert(doc, show_progress=True)
    else:
        # Create a VectorStoreIndex from the loaded documents
        documents = SimpleDirectoryReader(dir_from).load_data()
        index = VectorStoreIndex.from_documents(documents, show_progress=True)

    # Persist the updated index to storage
    index.storage_context.persist(persist_dir=persist_dir)

    # Move files from dir_from to dir_to
    for file_name in os.listdir(dir_from):
        file_path = os.path.join(dir_from, file_name)
        if os.path.isfile(file_path):
            shutil.move(file_path, dir_to)


def update_public_data(was_persisted=True):
    print("Updating public data...")
    add_to_index("./data/public_new", "./data/public", './data/public_persist', was_persisted)
    print("Public data update complete.")

def update_private_data(was_persisted=True):
    # Not implemented yet: private data is not ready to use

    print("Updating private data...")
    add_to_index("./data/private_new", "./data/private", './data/private_persist', was_persisted)
    print("Private data update complete.")


def test_query(index, query):
    # Create a query engine from the index
    query_engine = index.as_query_engine()

    # Perform a query using the query engine
    response = query_engine.query(query)
    print(response)
    return response


def test_retrieval(index, query):
    retriever = index.as_retriever(
        retriever_mode="llm",
        choice_batch_size=5,
    )

    response = retriever.retrieve(query)
    return response

def fetch_question_list(class_name=18650):
    """
    returns a list of questions from data/questions.json
    """
    import json
    with open(f'./data/questions/{str(class_name)}questions.json') as f:
        questions = json.load(f)
    return questions['questions']


# update_public_data(True)
# update_private_data(True)

# public_index = index_from_dir('./data/public_persist')
# retrieved = test_retrieval(public_index, 'Summarize the example of Slutsky\'s theorem that is given. Which lecture (1-3) is it from?')
# # print(retrieved)
# 
# print(type(retrieved))
# 
# for ele in retrieved:
#     print("-----")
#     print()
#     print(ele)

# test_query(public_index, "What is the lln, according to the lectures, and which lecture is it from?")