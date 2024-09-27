import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv

load_dotenv()

class Chain:
    def __init__(self):
        self.llm = ChatGroq(model="llama-3.1-70b-versatile",temperature=0,groq_api_key=os.getenv("GROQ_API_KEY"))

    def extract_jobs(self, cleaned_text):
        prompt_extract = PromptTemplate.from_template(
        """
        ###SCRAPED TEXT FROM WEBSITE
        {page_data}
        ### INSTRUCTION
        The scrapped text is from the career's page of a website.
        Your job is to extract the job posting and return it in json format containing the following keys: `role`, `experience`, `skill` and `description`.
        Return only valid json.
        ### VALID JSON(NO PREAMBLE):
        """
        ) 

        chain_extract=prompt_extract|self.llm
        res=chain_extract.invoke(input={'page_data': cleaned_text})

        try:
            json_parser = JsonOutputParser()
            res = json_parser.parse(res.content)
        except OutputParserException:
            raise OutputParserException("Content too big. Unable to parse jobs.")
        return res if isinstance(res, list) else [res]
    
    def write_mail(self, job, name, curr_job, curr_company):
        prompt_email = PromptTemplate.from_template(
            """
            ### JOB DESCRIPTION:
            {job_description}

            ### INSTRUCTION:
            You are {name_person}, a job seeker. You are skilled with the skills given in the job_description.   
            Your job is to write a cold email to the HR of the company regarding the job mentioned above describing the capability of yours 
            in fulfilling their needs.
            Remember you are {name_person}, {job_title} at {current_company}. 
            Do not provide a preamble.
            ### EMAIL (NO PREAMBLE):

            """
        )
        chain_email = prompt_email | self.llm
        res = chain_email.invoke({"job_description": str(job), "name_person": name, "job_title": curr_job, "current_company": curr_company})
        return res.content
    
if __name__ == "__main__":
    print(os.getenv("GROQ_API_KEY"))