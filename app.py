import asyncio
from gpt_researcher import GPTResearcher
import re
import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel, EmailStr, HttpUrl, Field, validator
from typing import List, Optional
from datetime import date, datetime
from prettytable import PrettyTable
from utilities import generate_search_queries, news_articles, articles, fetch_company_data
import pandas as pd
import base64
import io
import ast
import json
import markdown

load_dotenv()

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    openai_api_base="https://api.openai.com/v1",
    openai_api_key=st.secrets["OPENAI_API_KEY"],
    model_name="gpt-4o-mini",
)


class ContactInformation(BaseModel):
    email: Optional[str] = Field(None, description="Company email address")
    phone: Optional[str] = Field(None, description="Company phone number")
    website: Optional[str] = Field(None, description="Company website URL")

    @validator("email")
    def validate_email(cls, v):
        if v is None or v.strip() == "":
            return None
        if v and "@" in v and "." in v:
            return v
        return None

    @validator("website")
    def validate_website(cls, v):
        if v is None or v.strip() == "":
            return None
        if v and ("http://" in v or "https://" in v):
            return v
        return None


class CompanyInformation(BaseModel):
    primary_address: str = Field(
        ..., description="Complete registered address of the company"
    )
    registration_number: Optional[str] = Field(
        None, description="Company registration/identification number"
    )
    legal_form: str = Field(..., description="Legal structure of the company")
    country: str = Field(..., description="Country where company is registered")
    town: str = Field(..., description="City and state/province of registration")
    registration_date: str = Field(..., description="Date of company incorporation")
    contact_information: ContactInformation = Field(
        ..., description="Company contact details"
    )
    general_details: str = Field(..., description="Brief description of the company")
    ubo: Optional[str] = Field(None, description="Ultimate Business Owners information")
    directors_shareholders: List[str] = Field(
        ..., description="List of directors and shareholders"
    )
    subsidiaries: Optional[str] = Field(
        None, description="Information about company subsidiaries"
    )
    parent_company: Optional[str] = Field(
        None, description="Parent company information if any"
    )
    last_reported_revenue: str = Field(
        ..., description="Latest reported revenue information"
    )

    @validator("registration_number")
    def validate_registration_number(cls, v):
        if v is None or v.strip() == "":
            return "Information not available"
        # Remove common separators and clean up
        cleaned = re.sub(r"[^a-zA-Z0-9]", "", v)
        return cleaned if cleaned else "Information not available"

    @validator("directors_shareholders", pre=True)
    def parse_directors(cls, v):
        if isinstance(v, list):
            return v
        if isinstance(v, str):
            directors = [d.strip() for d in re.split(r"[,;]|\band\b", v) if d.strip()]
            return directors or ["Information not available"]
        return ["Information not available"]


def sanitize_string(s: str) -> str:
    if not s:
        return "Information not available"
    return s.strip() or "Information not available"


async def generate_evidence(query: str):
    try:
        researcher = GPTResearcher(
            query=query, report_type="research_report", config_path=None
        )
        await researcher.conduct_research()
        report = await researcher.write_report()

        split_text = report.split("## References")
        main_text = split_text[0].strip()

        if "## Conclusion" in main_text:
            main_text = main_text.split("## Conclusion")[0].strip()

        citation_pattern = r"\[(.*?)\]\((https?://\S+)\)"
        citations = re.findall(citation_pattern, main_text)

        references = "\n".join(
            [
                f"- {source.strip()} {url.strip().replace(')', '')}"
                for source, url in citations
            ]
        )

        main_text = re.sub(citation_pattern, "", main_text).strip()

        if len(split_text) > 1:
            references += "\n" + split_text[1].strip()

        return main_text, references
    except Exception as e:
        st.error(f"Error during research: {str(e)}")
        return "", "No references available due to error"


def format_company_data_as_dict(company_info):
    try:
        directors_str = ", ".join(company_info.directors_shareholders)

        all_fields = {
            "Fields": [
                "Primary Address",
                "Registration Number",
                "Legal Form",
                "Country",
                "Town",
                "Registration Date",
                "Email",
                "Phone",
                "Website",
                "General Details",
                "Directors & Shareholders",
                "UBO",
                "Subsidiaries",
                "Parent Company",
                "Last Reported Revenue",
            ],
            "Details": [
                sanitize_string(company_info.primary_address),
                sanitize_string(company_info.registration_number),
                sanitize_string(company_info.legal_form),
                sanitize_string(company_info.country),
                sanitize_string(company_info.town),
                sanitize_string(company_info.registration_date),
                sanitize_string(company_info.contact_information.email),
                sanitize_string(company_info.contact_information.phone),
                sanitize_string(str(company_info.contact_information.website)),
                sanitize_string(company_info.general_details),
                directors_str,
                sanitize_string(company_info.ubo),
                sanitize_string(company_info.subsidiaries),
                sanitize_string(company_info.parent_company),
                sanitize_string(company_info.last_reported_revenue),
            ],
        }
        return all_fields
    except Exception as e:
        st.error(f"Error formatting company data: {str(e)}")
        return {
            "Fields": ["Error"],
            "Details": ["Failed to format company information"],
        }


def final_output_generation(llm, report):
    try:
        structured_llm = llm.with_structured_output(CompanyInformation)
        result = structured_llm.invoke(report)
        return result
    except Exception as e:
        st.error(f"Error processing company information: {str(e)}")
        return CompanyInformation(
            primary_address="Information not available",
            registration_number="Information not available",
            legal_form="Information not available",
            country="Information not available",
            town="Information not available",
            registration_date=date(1900, 1, 1),
            contact_information=ContactInformation(),
            general_details="Information not available",
            directors_shareholders=["Information not available"],
            last_reported_revenue="Information not available",
        )
        
def reference_generation(llm, references):
    prompt = f"""
    Convert the given reference md file to a json representation with urls only, ignore all other data like title, author, etc, if present. Return JSON file only.
    References: {references}"""+"""
    OUTPUT FORMAT:
    {
        "References": ["url1", "url2", "url3"]
    }
    """
    response = llm.invoke(prompt)
    result = response.content
    result = result.replace("```json", "").replace("```", "")
    result = json.loads(result)
    return result

def convert_df_to_json(df, output_file_name):
    output = io.StringIO()
    df.to_json(output, orient="records", indent=4)
    return output.getvalue()

def get_download_link_json(json_data, file_name):
    json_bytes = json_data.encode()  # Convert str to bytes
    b64 = base64.b64encode(json_bytes).decode()  # Encode to base64
    href = f'<a href="data:application/json;base64,{b64}" download="{file_name}">Download Web Research Results</a>'
    return href

def convert_df_to_excel(df, output_file_name):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name=output_file_name)
    return output.getvalue()

def get_download_link_excel(excel_data, file_name):
    b64 = base64.b64encode(excel_data).decode()
    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{file_name}">Download Web Research Results</a>'
    return href

def get_analysis_results(content_list, company):
    prompt = f"""
    Analyse the following content and identify the key findings related to company, {company}, from the list provided. Return maximum 15 key findings as bullet points. Make sure that the key findings are unique and related to {company}. Do not include any other text other than the key findings.
    Content: {content_list}
    
    OUTPUT FORMAT:
    "- Key Finding 1\n
    - Key Finding 2"
    if key findings are found
    
    ""
    otherwise
    """
    response = llm.invoke(prompt)
    return response.content

def director_check(content, company, data_dict):
    prompt = f"""
From {data_dict}, for {company}, identify the directors. From the information, perform director sanity check on the provided content below. 
Return every content that refers to the directors of the company. From that content, analyse it and provide bullet points related to the directors only.
Do not include any other text other than the director check analysis. Return as bullet points for markdown file.
Content: {content}
OUTPUT FORMAT:
- Point 1
- Point 2
"""
    response = llm.invoke(prompt)
    return response.content

def analyze_sentiment_by_tag(df):
    def parse_tags(tag_str):
        try:
            if isinstance(tag_str, str):
                return ast.literal_eval(tag_str)
            return tag_str
        except (ValueError, SyntaxError):
            return []
    
    processed_df = df.copy()
    processed_df['parsed_tags'] = processed_df['tags'].apply(parse_tags)
    
    tag_counts = {}
    
    for _, row in processed_df.iterrows():
        tags = row['parsed_tags']
        sentiment = row['sentiment']
        
        if not isinstance(tags, list) or len(tags) == 0:
            continue
        
        for tag in tags:
            tag = tag.capitalize()
            if tag not in tag_counts:
                tag_counts[tag] = {
                    'total': 0,
                    'Positive': 0,
                    'Negative': 0,
                    'Neutral': 0
                }
            
            tag_counts[tag]['total'] += 1
            tag_counts[tag][sentiment] += 1
    
    result_rows = []
    for tag, counts in tag_counts.items():
        total = counts['total']
        row_data = {'tag': tag, 'total_articles': total}
        
        for sentiment in ['Positive', 'Negative', 'Neutral']:
            count = counts.get(sentiment, 0)
            percentage = (count / total) * 100 if total > 0 else 0
            row_data[sentiment] = round(percentage, 2)
        
        result_rows.append(row_data)
    
    if not result_rows:
        return pd.DataFrame(columns=['Negative', 'Neutral', 'Positive', 'total_articles'])
    
    result_df = pd.DataFrame(result_rows)
    result_df = result_df.set_index('tag')
    result_df = result_df[['Negative', 'Neutral', 'Positive', 'total_articles']]
    result_df = result_df.drop(columns=['total_articles'])
    
    return result_df

def md_to_html(input_md, output_html):
    with open(input_md, "r", encoding="utf-8") as file:
        md_content = file.read()

    html_content = markdown.markdown(md_content, extensions=["extra"])

    with open(output_html, "w", encoding="utf-8") as file:
        file.write(html_content)
    
def main():
    st.title("AI Web Research Agent")

    company_name = st.text_input("Enter the name of the company:")
    col2, col3 = st.columns(2)
    
    with col2:
        country = st.text_input("Enter country (optional):")
        
    with col3:
        state = st.text_input("Enter state (optional):")
        

    fromDate = st.date_input("Start Date", date.today())
    toDate = st.date_input("End Date", date.today())

    if fromDate > toDate:
        st.error("Start date must be before end date")

    if st.button("Fetch Details"):
        if not company_name:
            st.warning("Please enter a company name.")
            return
        
            
        max_results = 2
        try:
            with st.spinner("Fetching details, please wait..."):
                if country.strip() and state.strip():
                    query = f"Provide the {company_name} details specifically in {state},{country}, including registration number, primary address, legal form, country, town, registration date, contact info, general details, UBO, directors/shareholders, subsidiaries, parent company, and last reported revenue. Focus on the company's operations, registration number, and registration in {country}."
                elif country.strip() and not state.strip():
                    query = f"Provide the {company_name} details specifically in {country}, including registration number, primary address, legal form, country, town, registration date, contact info, general details, UBO, directors/shareholders, subsidiaries, parent company, and last reported revenue. Focus on the company's operations, registration number, and registration in {country}."
                elif state.strip() and not country.strip():
                    query = f"Provide the {company_name} details specifically in {state}, including registration number, primary address, legal form, country, town, registration date, contact info, general details, UBO, directors/shareholders, subsidiaries, parent company, and last reported revenue. Focus on the company's operations, registration number, and registration in {country}."
                else:
                    query = f"Provide the {company_name} details, including registration number, primary address, legal form, country, town, registration date, contact info, general details, UBO, directors/shareholders, subsidiaries, parent company, and last reported revenue. Make sure to include any company registration or identification numbers."

                report, references = asyncio.run(generate_evidence(query=query))
                result = final_output_generation(llm, report)
                ref_dict = reference_generation(llm, references)

                data_dict = format_company_data_as_dict(result)
                
                combined_data = {"Company_Data": data_dict, "References": ref_dict}
                with open(f"{company_name}_company_research.json", "w") as file:
                    json.dump(combined_data, file, indent=4)

                st.subheader("AI Web Research Agent Result: ")
                st.table(data_dict)

                st.subheader("References")
                st.markdown(references, unsafe_allow_html=True)
                
                if not country:
                    country = ""
                queries = generate_search_queries(company_name, country, data_dict)
                df = pd.DataFrame(columns=["url", "content", "sentiment", "tags"])
                corporate_actions = queries["corporate_actions"]
                adverse_media = queries["adverse_media"]
                df = news_articles(queries["search_queries"], df, company_name, corporate_actions, adverse_media, fromDate, toDate, max_results)
                df_articles = articles(company_name, corporate_actions, adverse_media, max_results, fromDate, toDate)
                df = pd.concat([df, df_articles], ignore_index=True)
                output_file_name_json = f"{company_name}_web_research_results.json"
                output_file_name_excel = f"{company_name[:4]}_web_research_results.xlsx"
                sentiment_counts = df["sentiment"].value_counts().reset_index()
                sentiment_counts.columns = ["Sentiment", "Count"]
                positive_content = df[df['sentiment'] == 'Positive']['content'].tolist()
                negative_content = df[df['sentiment'] == 'Negative']['content'].tolist()
                neutral_content = df[df['sentiment'] == 'Neutral']['content'].tolist()
                
                positive_content = get_analysis_results(positive_content, company_name)
                negative_content = get_analysis_results(negative_content, company_name)
                neutral_content = get_analysis_results(neutral_content, company_name)
                
                content_list = df["content"].tolist()
                director_content = director_check(content_list, company_name, data_dict)
                sent_df = analyze_sentiment_by_tag(df)
                
                with open(f"{company_name}.md", "w") as f:
                    f.write("# Adverse Media Research Results\n")
                    md_table = sentiment_counts.to_markdown(index=False)
                    f.write(md_table)
                    if positive_content != '""':
                        f.write("\n\n## Positive Media Keypoints:\n")
                        f.write(positive_content)
                
                    if negative_content!= '""':
                        f.write("\n\n## Negative Media Keypoints:\n")
                        f.write(negative_content)
                
                    if neutral_content!= '""':
                        f.write("\n\n## Neutral Media Keypoints:\n")
                        f.write(neutral_content)
                        
                    f.write("\n\n## Sentiment Distribution by Category\n")
                    md_table2 = sent_df.to_markdown(index=False)
                    f.write(md_table2)
                                    
                    f.write("\n\n## Directors Sanity Check\n")
                    f.write(director_content)
                    
                md_to_html(f"{company_name}.md", f"{company_name}.html")

                json_data = convert_df_to_json(df, output_file_name_json)
                st.markdown(get_download_link_json(json_data, output_file_name_json), unsafe_allow_html=True)
                json_df = df.to_json(output_file_name_json, orient="records", indent=4)
                
                excel_data = convert_df_to_excel(df, output_file_name_excel)
                st.markdown(get_download_link_excel(excel_data, output_file_name_excel), unsafe_allow_html=True)
                excel_df = df.to_excel(output_file_name_excel, index=False)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.warning(
                "Please try again with a different company name or check your internet connection."
            )

if __name__ == "__main__":
    main()
