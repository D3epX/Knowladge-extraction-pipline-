import lancedb


#==========================================================================
#          database connection 
#==========================================================================

DB_PATH = "/home/dahmane/dev/Knowledge-Extraction-Pipeline/src/data/lancedb"
db = lancedb.connect(DB_PATH)

#==========================================================================
#                 loading table
#==========================================================================
table = db.open_table("docling")

#==========================================================================
#                 search table
#==========================================================================
result = table.search(query="pdf").limit(5).to_pandas() #embedding-enabled text search; LanceDB will vectorize the query automatically.and it uses similarity search to find the most relevant chunks based on their embeddings then return top5 
print(result)