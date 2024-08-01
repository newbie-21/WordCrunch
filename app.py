import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Model and tokenizer for summarization
tokenizer = AutoTokenizer.from_pretrained("suriya7/bart-finetuned-text-summarization")
model = AutoModelForSeq2SeqLM.from_pretrained("suriya7/bart-finetuned-text-summarization")


def summarization(text, max_length, min_length):
    tokenized_text = tokenizer.encode(text, return_tensors='pt', max_length=512, truncation=True)
    summary_ids = model.generate(
        tokenized_text,
        num_beams=8,
        min_length=min_length,  # Ensure summary is at least this long
        max_length=max_length,  # Ensure summary is no longer than this
        early_stopping=True
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary


def summarize_long_text(text, max_length, min_length, chunk_size=1000):
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    chunk_summaries = [summarization(chunk, max_length, min_length) for chunk in chunks]
    combined_summary = " ".join(chunk_summaries)

    # Final adjustment to ensure the summary meets the length requirements
    final_summary = summarization(combined_summary, max_length, min_length)
    return final_summary


def summarize_text(text, max_length, min_length):
    summary = summarization(text, max_length, min_length)
    if len(summary) > max_length or len(summary) < min_length:
        summary = summarize_long_text(summary, max_length, min_length)
    return summary


# Initialize session state
if 'input_text' not in st.session_state:
    st.session_state.input_text = ""
if 'max_length' not in st.session_state:
    st.session_state.max_length = 512
if 'min_length' not in st.session_state:
    st.session_state.min_length = 30
if 'summary' not in st.session_state:
    st.session_state.summary = ""

# Streamlit UI
st.title("WordCrunch")

# Input fields
st.session_state.input_text = st.text_area("Enter Text", height=200, value=st.session_state.input_text)
st.session_state.max_length = st.number_input("Enter the maximum output size", min_value=10, max_value=2000, value=st.session_state.max_length)
st.session_state.min_length = st.number_input("Enter the minimum output size", min_value=1, max_value=1000, value=st.session_state.min_length)

# Buttons in columns
col1, col2 = st.columns([7, 1])

with col1:
    if st.button("Summarize"):
        if st.session_state.input_text:
            with st.spinner("Generating summary..."):
                st.session_state.summary = summarize_text(st.session_state.input_text, st.session_state.max_length, st.session_state.min_length)
        else:
            st.warning("Please enter some text.")

with col2:
    if st.button("Reset"):
        st.session_state.input_text = ""
        st.session_state.max_length = 512
        st.session_state.min_length = 30
        st.session_state.summary = ""

if st.session_state.summary:
    st.write("**Summary:**")
    st.write(st.session_state.summary)
