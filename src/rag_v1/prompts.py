class Prompts:
    answer_prompt_en = """You are answering the user's visual question.
    
You are about to be given the user's question, the retrieved web context and visual inputs. You need to answer the question based on the context and visual inputs.
    
Use the web context if it is helpful.
You should identify the user's requirements based on the question first. If the image or some images are completely irrelated to the questions, please ignore them. Avoid forcibly associating unrelated images with the issue.

The user question is:
"{question}"

Web context:
{context}
"""

    web_prompt_en_backup = """You are building a web search query for a visual RAG pipeline.

Look at the image or images and the user question, then produce one concise web search query.

The user question is:
"{question}"

Your query should meet the following requirements:
1. The user's question has the highest priority. Identify the parts of the image that are relevant to the user's question and use them to construct the query.
2. If the image content is completely unrelated to the user's question, or if the image is too unclear to interpret, prioritize constructing the query based on the user's question alone, ignoring any irrelevant or unrecognizable elements in the image.
3. If the user's question is closely related to the image or images, the query should include the most visible entities, OCR text if any, landmarks, brands, materials, or domain terms that help retrieval.
4. Do not answer the question directly.
5. Do not add explanations, quotes, numbering, or multiple options.
6. Output only a single line search query.
"""

    web_prompt_en = '''You are building a web search query for a visual RAG pipeline.
    
You are about to be given the user's question and one or more images.

The user question is:
"{question}"

You should construct the query for web retrieval based on the following requirments.

## Overall data format requirements:

    You should output only a single line search query.
    
## Content requirements:
    
    1. The user's question has the highest priority. The generated query should be relevant to the user's question, and the search results should be helpful in answering the question.
    2. The query should be directly searchable by the browser. Therefore, it should be concise, composed of keywords, not complete sentences, and avoid unnecessary punctuation.
    3. Do not answer the question directly.
    4. Do not add any explanations, quotes, numbering, or multiple options.

## Construction guidelines:

    You should construct the query based on the following steps:

    ** 1. Understand the question:**
        Identify the user's requirements based on the question. If the user's question is very simple, or a question about basic common sense, or if the user explicitly asks you not to refer to the image or images, you can directly construct the query based on the user's question and skip the following steps.

    
    ** 2. Understanding the image or images:**
        Identify the parts of the image that are relevant or helpful to the user's question. If the content of some images is completely unrelated to the user's question, or if some images are too unclear to interpret, prioritize constructing the query based on the user's question alone, ignoring any irrelevant or unrecognizable elements in the image.
        
        If the user's question is closely related to the image or images, the query should include the most visible entities, OCR text if any, landmarks, brands, materials, or domain terms that help retrieval.
        
    ** 3. Construct query: **
        Based on the information you get above, construct the query.

## Examples:

    **Example 1:**
        Input:
            image description: "A photo of Donald Trump."
            user question: "Has he recently issued any new policies?"
        Output:
            Donald Trump new policies announced recently

    **Example 2:**
        Input:
            image description: "A photo of Huangshan Welcoming Pine."
            user question: "Where was this photo taken?"
        Output:
            Huangshan geographical location
            
    **Example 3:**
        Input:
            image description: "A photo of a tall building."
            user question: "What were Yuan Longping's outstanding contributions?"
        Output:
            Yuan Longping achievements
'''

    freshness_prompt_en = """You are choosing a web-search freshness parameter for a visual RAG pipeline. This parameter controls the time range for the network search.

Look at the image or images, the user's question, the generated search query, and the current time, then choose exactly one freshness value.

Available freshness values:
oneDay: results from the past day
oneWeek: results from the past week
oneMonth: results from the past month
oneYear: results from the past year
noLimit: no time limit

Search query:
{query}

User question:
{question}

Current time:
{current_time}

Selection guidance:
1. If the user explicitly specifies a time range, strictly follow the user's requirements.
2. Use "oneDay" for breaking news, live events, today/yesterday, current prices, current schedules, or very time-sensitive facts.
3. Use "oneWeek" for recent events, newly released products, recent announcements, or questions likely to change within days.
4. Use "oneMonth" for recent but not daily-changing topics.
5. Use "oneYear" for topics where information from the past year is enough and older content may be stale.
6. Use "noLimit" for stable topics such as landmarks, historical facts, general explanations, definitions, or timeless visual identification.
7. Return only one of: oneDay, oneWeek, oneMonth, oneYear, noLimit.
"""

    wikipedia_prompt_en = """You are building a Wikipedia search query for a visual RAG pipeline.

Look at the image or images and the user question, then produce one concise wikipedia search query.

The user question is:
"{question}"

Your query should meet the following requirements:
1. Your query should not be too complex, otherwise Wikipedia will not be able to match it.
2. If some of the images are completely unrelated to the user's question, please ignore them. Generate the query based solely on the text if all images are completely unrelated to the user's quesiton.
3. If you are unsure what the object in the image is, you don't need to provide a very precise query; instead, provide a general query.
4. Do not answer the question directly.
5. Do not add explanations, quotes, numbering, or multiple options.
6. Return only the search query.
"""


prompts = Prompts()
