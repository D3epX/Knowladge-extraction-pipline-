from docling.document_converter import DocumentConverter 
from utils.sitemap import get_sitemap_urls

converter = DocumentConverter()
#======================================================================
#            BASIC PDF EXTRACTION
#======================================================================

'''source  ="/home/dahmane/dev/Knowledge-Extraction-Pipeline/data/docling.pdf"

doc = converter.convert(source).document
markdown_output = doc.export_to_markdown()



with open("output.md", "w")as f :
    f.write(markdown_output)
f.close()'''

#======================================================================
#             Scrape multiple pages using the sitemap
#======================================================================

sitemap_url = get_sitemap_urls("https://www.samsung.com/")

conv_results_iter = converter.convert_all(sitemap_url)
docs =[]
for result in conv_results_iter:
    if result.document:
        document = result.document
        docs.append(document)

print(docs[0].export_to_markdown())

