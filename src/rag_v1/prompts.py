class Prompts:
    answer_prompt_en = """Answer the user's question based on the provided web context and the image or images.
Use the web context if it is helpful.
You should stay consistent with what is visible in the image. However, if the image or images are completely irrelated to the questions, please ignore them.
If the context is insufficient or uncertain, say so clearly.

Web Context:
{context}

Question:
{question}
"""

    direct_answer_prompt_en = """You are about to be given a image or images.
You need to output "Yes" or "No", reflecting whether these images are sufficient to answer this question: "{question}"

Your output should meet the following requirements:
1. If the image or images are completely unrelated to the question, and you can
"""

    web_prompt_en = """You are building a web search query for a visual RAG pipeline.

Look at the image or images and the user question, then produce one concise web search query.

The user question is:
"{question}"

Your query should meet the following requirements:
1. The query should include the most important visible entities, OCR text if any, landmarks, brands, materials, or domain terms that help retrieval.
2. Some of the images can be completely unrelated to the user's question, please ignore them. Generate the query based solely on the text if all images are completely unrelated to the user's quesiton.
3. Do not answer the question directly.
4. Do not add explanations, quotes, numbering, or multiple options.
5. Return only the search query.

User question:
{question}
"""

    freshness_prompt_en = """You are choosing a web-search freshness parameter for a visual RAG pipeline.

Look at the image or images, the search query, and the current time, then choose exactly one freshness value.

Available freshness values:
oneDay: results from the past day
oneWeek: results from the past week
oneMonth: results from the past month
oneYear: results from the past year
noLimit: no time limit

Current time:
{current_time}

Search query:
{query}

Selection guidance:
1. Use "oneDay" for breaking news, live events, today/yesterday, current prices, current schedules, or very time-sensitive facts.
2. Use "oneWeek" for recent events, newly released products, recent announcements, or questions likely to change within days.
3. Use "oneMonth" for recent but not daily-changing topics.
4. Use "oneYear" for topics where information from the past year is enough and older content may be stale.
5. Use "noLimit" for stable topics such as landmarks, historical facts, general explanations, definitions, or timeless visual identification.
6. Return only one of: oneDay, oneWeek, oneMonth, oneYear, noLimit.
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
