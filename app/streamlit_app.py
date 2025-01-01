__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
from streamlit_pdf_viewer import pdf_viewer
from functions import *
import base64



    
def display_pdf(uploaded_file):
    # Read file as bytes
    bytes_data = uploaded_file.getvalue()
    
    # Convert to Base64
    base64_pdf = base64.b64encode(bytes_data).decode('utf-8')  # Correct encoding
    
    # Embed PDF in HTML
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
    
    # Display file
    st.markdown(pdf_display, unsafe_allow_html=True)
    

def load_streamlit_page():
    st.set_page_config(layout="wide", page_title="LLM Tool")
    
    col1, col2 = st.columns([0.5, 0.5], gap="large")
    
    with col1:
        st.header("Input your OpenAI API key")
        st.text_input('OpenAI API key', type='password', key='api_key',
                      label_visibility="collapsed", disabled=False)
        st.header("Upload file")
        upload_file = st.file_uploader("Upload a PDF", type="pdf")
        # upload_file = st.file_uploader("Please upload your PDF document:", type="pdf")
        
        # upload_file = st.file_uploader("Upload a PDF", type="pdf")
    return col1, col2, upload_file
    
    
# Make a Streamlit page
col1, col2, upload_file = load_streamlit_page()

# Load CSS file
# def load_css(file_name):
#     with open(file_name) as f:
#         st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def load_css(file_name):
    # Build the absolute path to the CSS file
    file_path = os.path.join(os.path.dirname(__file__), file_name)
    try:
        with open(file_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"CSS file '{file_name}' not found at path: {file_path}")

# # Call the function to load styles.css
load_css("static/styles.css")

# Debugging working directory and file structure
# st.write("Current working directory:", os.getcwd())
# st.write("Files in current directory:", os.listdir(os.getcwd()))

def display_pdf_with_viewer(uploaded_file):
    """Displays a PDF using the streamlit-pdf-viewer component."""
    bytes_data = uploaded_file.getvalue()
    pdf_viewer(bytes_data, height=600, width=500)

# Streamlit app
# 

# uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

# if uploaded_file:
#     st.write("PDF Preview:")
#     display_pdf_with_viewer(uploaded_file)
    

if 'api_key' not in st.session_state:
    st.session_state.api_key = ''
    
# # Process the input
if upload_file is not None:
    with col2:
        st.write("PDF Preview")
        # display_pdf(upload_file)
        display_pdf_with_viewer(upload_file)
        
    # Load in the documents
    documents = get_pdf_text(upload_file)
    st.session_state.vector_state = create_vectorstore_from_texts(documents,
                                                                  api_key=st.session_state.api_key,
                                                                  file_name=upload_file.name)
    
    # st.write('Input Processed')



# Generate answer
with col1:
    if st.button("Generate table"):
        with st.spinner("Generating answer"):
            # Load vectorstore:
            st.title("PDF Extraction")
            answer = query_document(vectorstore=st.session_state.vector_state,
                                    query="What is the title of the research article.",
                                    api_key=st.session_state.api_key)
            
            placeholder = st.empty()
            placeholder.write(answer)


# Footer
st.markdown(
    """
    <div class="footer">
        Langchain extracting unstructured PDF data — <a href="https://github.com/alimuddinmllg/Langchain-unstructured-extraction" target="_blank" style="color: #fff; text-decoration: underline;">Alimuddin Melleng</a>
    </div>
    """,
    unsafe_allow_html=True
)
