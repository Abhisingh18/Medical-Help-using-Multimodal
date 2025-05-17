import streamlit as st
import base64
import requests

# API key secure tarike se Streamlit secrets se lein
GROK_API_KEY = st.secrets["GROK_API_KEY"]

GROK_API_URL = "https://api.x.ai/v1/chat/completions"

sample_prompt = """You are a medical practitioner and an expert in analyzing medical-related images working for a very reputed hospital. You will be provided with images and you need to identify the anomalies, any disease or health issues. You need to generate the result in a detailed manner. Write all the findings, next steps, recommendation, etc. You only need to respond if the image is related to a human body and health issues. You must have to answer but also write a disclaimer saying that "Consult with a Doctor before making any decisions".

Remember, if certain aspects are not clear from the image, it's okay to state 'Unable to determine based on the provided image.'

Now analyze the image and answer the above questions in the same structured manner defined above."""

if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'result' not in st.session_state:
    st.session_state.result = None

def encode_image_bytes(uploaded_file):
    file_bytes = uploaded_file.read()
    return base64.b64encode(file_bytes).decode('utf-8')

def call_grok_model_for_analysis(uploaded_file, sample_prompt=sample_prompt):
    base64_image = encode_image_bytes(uploaded_file)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": sample_prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": "high"
                    }
                }
            ]
        }
    ]
    headers = {
        "Authorization": f"Bearer {GROK_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "grok-1.5",
        "messages": messages,
        "max_tokens": 1500
    }
    response = requests.post(GROK_API_URL, headers=headers, json=data)
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        return f"Error: {response.status_code} - {response.text}"

def chat_eli(query):
    eli5_prompt = "You have to explain the below piece of information to a five years old:\n" + query
    messages = [
        {
            "role": "user",
            "content": eli5_prompt
        }
    ]
    headers = {
        "Authorization": f"Bearer {GROK_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "grok-1.5",
        "messages": messages,
        "max_tokens": 1500
    }
    response = requests.post(GROK_API_URL, headers=headers, json=data)
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        return f"Error: {response.status_code} - {response.text}"

# Streamlit UI
st.title("🧠 Medical Help using Multimodal LLM (Grok xAI)")

with st.expander("ℹ️ About this App"):
    st.write("Upload a medical-related image to get an AI-based analysis using Grok (xAI) with vision capabilities.")

uploaded_file = st.file_uploader("📤 Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.session_state['uploaded_file'] = uploaded_file
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

if st.button('🔍 Analyze Image'):
    if st.session_state['uploaded_file'] is not None:
        with st.spinner("Analyzing the image with Grok..."):
            st.session_state['result'] = call_grok_model_for_analysis(st.session_state['uploaded_file'])
        st.markdown(st.session_state['result'], unsafe_allow_html=True)

if st.session_state.get('result'):
    st.info("Want a simple explanation? Try ELI5 👶")
    if st.radio("Explain Like I'm 5 (ELI5)?", ('No', 'Yes')) == 'Yes':
        with st.spinner("Simplifying..."):
            simplified_explanation = chat_eli(st.session_state['result'])
        st.markdown(simplified_explanation, unsafe_allow_html=True)
