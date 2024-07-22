import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv


load_dotenv()


os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")


llm = ChatGroq(model="llama3-8b-8192")


prompt_template = PromptTemplate(
    template="Generate a {language} code snippet for the following task:\n{task}\n",
    input_variables=["language", "task"]
)


code_generation_chain = LLMChain(
    llm=llm,
    prompt=prompt_template
)


def generate_code(language, task):
    input_data = {"language": language, "task": task}
    response = code_generation_chain.run(input_data)
    return response


def main():
    st.title("Code  Generator")
    language = st.selectbox("Select a programming language", ["Python", "JavaScript", "Java", "C++", "Go", "Ruby", "PHP"])
    task = st.text_area("Describe the task you want the code for")

    if st.button("Generate Code"):
        with st.spinner("Generating code..."):
            generated_code = generate_code(language, task)
            st.code(generated_code, language=language.lower())

if __name__ == "__main__":
    main()
