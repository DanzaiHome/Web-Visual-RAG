class Prompts:
    answer_prompt_en = """You are answering the user's visual question.

You are about to be given the user's question, the retrieved web context and visual inputs. You need to answer the question based on the context and visual inputs.

Use the web context if it is helpful.
You should identify the user's requirements based on the question first. If the image or some images are completely unrelated to the questions, please ignore them. Avoid forcibly associating unrelated images with the issue.

Evidence and temporal verification rules:
1. First identify the visual entity, then identify the target entity asked about. Use evidence only if it actually refers to that target entity.
2. If the question asks for a latest, most recent, current, today/yesterday, recent factual value, final score, result, latest release, current price, current schedule, or similar time-sensitive fact, do not assume the top-ranked document or the only related result is the latest/current event.
3. For latest/current questions, check whether the evidence contains:
   - the target entity,
   - the requested fact value,
   - an event date/time or other clear wording that proves the event is the latest/current/completed one.
4. Distinguish page metadata from event time. A document's Published, Last crawled, or date-like URL field is page/search metadata, not necessarily the date of the game, release, price quote, or event. Do not cite page metadata as proof that an event is the latest/current one.
5. When multiple candidate events or results are present, compare event dates from titles/content and use the most recent completed/occurred event, not merely the freshest webpage.
6. Use the pipeline current time, when provided, to reason about whether a scheduled event may have already occurred. If the context contains a later scheduled/preview event for the target entity that is before the pipeline current time, but no final score/result for that later event, do not fall back to an older completed result as the "latest"; say the latest result cannot be confirmed from the evidence.
7. For sports scores/results, prefer evidence that explicitly says final score, result, completed game, box score, schedule result, or an official/statistical source page. Do not treat a single old-looking score as the most recent game unless the evidence proves it. Do not use a document with an older event date as support for the current latest event.
8. If the context gives a relevant fact value but does not prove it is the latest/current one, answer conservatively: state the value found and say that the evidence is insufficient to confirm it is the latest/current result. If the only support for recency is page metadata, say that it appears relevant but is not independently confirmed as the latest/current event.
9. If answering with a score, price, count, date, or other numeric fact, briefly mention the supporting document title or source.

The user question is:
"{question}"

The retrieved context is:
{context}

Please answer the question directly. If the evidence is insufficient, say so explicitly instead of over-claiming.
"""

    web_prompt_en = '''You are building a web search query for a visual RAG pipeline.

You are about to be given the user's question and one or more images.

The user question is:
"{question}"

You should construct the query for web retrieval based on the following requirements:

## Overall data format requirements:

    You should output only a single line search query.

## Content requirements:

    1. The user's question has the highest priority. The generated query should be relevant to the user's question, and the search results should be helpful in answering the question.
    2. The query should be directly searchable by the browser. Therefore, it should be concise, composed of keywords, not complete sentences, and avoid unnecessary punctuation.
    3. Do not answer the question directly.
    4. Do not add any explanations, quotes, numbering, or multiple options.

## Construction guidelines:

    ** 1. Understand the question:**
        Identify the user's requirements based on the question. If the user's question is very simple, or a question about basic common sense, or if the user explicitly asks you not to refer to the image or images, you can directly construct the query based on the user's question and skip the following steps.

    ** 2. Understanding the image or images:**
        Identify the parts of the image that are relevant or helpful to the user's question. If the content of some images is completely unrelated to the user's question, or if some images are too unclear to interpret, prioritize constructing the query based on the user's question alone, ignoring any irrelevant or unrecognizable elements in the image.

        If the user's question is closely related to the image or images, the query should include the most visible entities, OCR text if any, landmarks, brands, materials, or domain terms that help retrieval.

    ** 3. Handle current/latest factoid questions:**
        If the question asks for a recent, latest, current, today/yesterday, last, most recent, final score, result, latest release, current price, current schedule, or similar time-sensitive factual value, build a query aimed at proving the current/latest fact instead of finding any related article.

        The query should include:
        - the target entity identified from the image and question,
        - the event or fact type being requested,
        - freshness/completion words such as latest, most recent, last, current, today, yesterday, completed, final score, result, official, or source-specific equivalents,
        - structured result page terms when useful, such as schedule, results, box score, standings, price, quote, market data, releases, official site, ESPN, NBA.com, Flashscore, Reuters, SEC filing, product page, or other relevant data-source terms.

        For current/latest factoid questions, the query must include at least one recency/completion term and at least one result-page/source term. For sports final-score questions, include the team/player entity plus latest/last, completed, final score, schedule, results, box score, or equivalent Chinese terms such as 最近一场, 已结束, 正式比赛, 最终比分, 赛程, 结果. Do not generate a vague query like "latest game score" when the user needs the most recent completed event.

    ** 4. Construct query: **
        Based on the information you get above, construct the query.

## Examples:

    **Example 1:**
        Input:
            image description: "The image shows the White House."
            user question: "Who lives here?"
        Output:
            White House current resident

    **Example 2:**
        Input:
            image description: "A mountain range that looks like Huangshan."
            user question: "Where was this photo taken?"
        Output:
            Huangshan geographical location

    **Example 3:**
        Input:
            image description: "A photo of a tall building."
            user question: "What were Yuan Longping's outstanding contributions?"
        Output:
            Yuan Longping achievements

    **Example 4:**
        Input:
            image description: "LeBron James wearing a Los Angeles Lakers jersey."
            user question: "What was the final score of the most recent official game for the team this person plays for, and how many points did that team score?"
        Output:
            Los Angeles Lakers latest completed game final score schedule results

    **Example 5:**
        Input:
            image description: "A public company logo on a building."
            user question: "What is its current stock price?"
        Output:
            company name current stock price quote market data official

    **Example 6:**
        Input:
            image description: "LeBron James wearing a Los Angeles Lakers jersey."
            user question: "图中这个人所在的球队最近一场正式比赛的最终比分是多少？这支球队得了多少分？"
        Output:
            洛杉矶湖人 最近一场 已结束 正式比赛 最终比分 赛程 结果
'''

    freshness_prompt_en = """You are choosing a web-search freshness parameter for a visual RAG pipeline. This parameter controls the time range for the network search.

The user question is:
{question}

The generated search query is:
{query}

Current time:
{current_time}

Selection guidance:
1. If the user explicitly specifies a time range, strictly follow the user's requirements.
2. Use "oneDay" for breaking news, live events, today/yesterday, current prices, current schedules, or very time-sensitive facts.
3. If the original user question asks for a latest/current/recent factual value, prefer a fresh range even if the generated query is broad. Use "oneDay" for current prices, latest scores/results, today's data, yesterday's data, current schedules, or latest completed events that can change daily.
4. Use "oneWeek" for recent events, newly released products, recent announcements, latest releases, most recent completed games/events, or questions likely to change within days when "oneDay" may be too narrow.
5. For "most recent game", "last completed match", "latest final score", or similar completed-event questions, use the original user question and current time to choose "oneDay" or "oneWeek"; do not choose a broad range only because the query contains stable entity names.
6. Use "oneMonth" for recent but not daily-changing topics.
7. Use "oneYear" for topics where information from the past year is enough and older content may be stale.
8. Use "noLimit" for stable topics such as landmarks, historical facts, general explanations, definitions, or timeless visual identification.
9. Return only one of: oneDay, oneWeek, oneMonth, oneYear, noLimit.
"""

    sufficiency_prompt_en = '''You are helping with a visual RAG QA progress.

You are about to be given the user's question, the user's visual input (image or images), and the retrieved context we have so far.

You need to judge whether the retrieved context is sufficient to answer the user the question.

User question:
{question}

Retrieved context:
{context}

Your response should meet the overall data format requirements:
    You need to provide a JSON object that follows the structure below.
    "{{
        "judgement": "...",
        "addition": "..."
    }}"
    Where "judgement" must be "YES" or "NO". If the "judgement" is True, the "addition" should be None. Else, "addition" should be a single line search query used to retrieve the missing information afterwards.

You should generate output based on the following steps:

** 1. Understand the question:**
    Identify the user's requirements based on the question. If the user's question is very simple, or a question about basic common sense, or if the user explicitly asks you not to refer to the image or images, you should output the JSON object whose "judgement" is "True" and skip all the steps afterwards.

** 2. Understanding the image or images:**
    Identify the parts of the image that are relevant or helpful to the user's question. If the content of some images is completely unrelated to the user's question, or if some images are too unclear to interpret, ignore any irrelevant or unrecognizable elements in the image.

    If the user's question is closely related to some images, refer to the retrieved context to see if you have enough information to understand the images clearly. If yes for all related images, output the JSON object whose "judgement" is "True" and skip all the steps afterwards.

** 3. Construct addition:**
    If you can not understand some of the related images, generate a single-line web browser query to fetch the information you lack, and put the query you generate in the JSON output.

    Try to fetch all the lack information you need with one query. If you can not fetch everything with one query, ONLY put one query in the output, do not put all of them in the output.
'''

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
