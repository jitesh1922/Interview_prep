## document chinking 
loader=TextLoader("speech.txt")
documents=loader.load()
text_splitter=CharacterTextSplitter(chunk_size=1000,chunk_overlap=30)
docs=text_splitter.split_documents(documents)
## embedding
embeddings=OllamaEmbeddings()
db=FAISS.from_documents(docs,embeddings)
### querying 
query="How does the speaker describe the desired outcome of the war?"
docs=db.similarity_search(query)
docs[0].page_content
## retriever
retriever=db.as_retriever()
docs=retriever.invoke(query)
docs[0].page_content
## similarity seach with score
docs_and_score=db.similarity_search_with_score(query)
docs_and_score
## similarity  embed query
embedding_vector=embeddings.embed_query(query)
##  similarity earch with vector 
docs_score=db.similarity_search_by_vector(embedding_vector)
## promt
system_prompt = (
                "You are an assistant for question-answering tasks. "
                "Use the following pieces of retrieved context to answer "
                "the question. If you don't know the answer, say that you "
                "don't know. Use three sentences maximum and keep the "
                "answer concise."
                "\n\n"
                "{context}"
            )
contextualize_q_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", contextualize_q_system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            )

## webbased loader 
loader=WebBaseLoader("https://docs.smith.langchain.com/")
docs=loader.load()
documents=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200).split_documents(docs)
vectordb=FAISS.from_documents(documents,OpenAIEmbeddings())
retriever=vectordb.as_retriever()
retriever