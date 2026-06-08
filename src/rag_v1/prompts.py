class Prompts:
    answer_prompt_en = """You are answering the user's visual question.

You are about to be given the user's question, the retrieved web context and visual inputs. You need to answer the question based on the evidence and visual inputs.

Citation and evidence grounding rules:
1. Retrieved evidence is organized as source blocks named [Doc 1], [Doc 2], etc.
2. Every factual claim that depends on retrieved web evidence must include one or more citations in square brackets, for example: [Doc 1] or [Doc 1][Doc 3].
3. A citation may only be used when that document actually supports the claim. Do not cite a source for a fact that is absent from that source.
4. Do not invent document ids, URLs, dates, names, scores, or quotations. Use only the documents present in the retrieved context.
5. If the retrieved context contains no [Doc n] blocks, answer from the image/question only and do not invent citations.

Temporal verification rules:
1. First identify the visual entity, then identify the target entity asked about. Use evidence only if it actually refers to that target entity.
2. For latest/current questions, check whether the evidence contains:
   - the target entity,
   - the requested fact value,
   - an event date/time or other clear wording that proves the event is the latest/current/completed one.
3. Distinguish page metadata from event time. A document's Published, Last crawled, or date-like URL field is page/search metadata, not necessarily the date of the game, release, price quote, or event. Do not cite page metadata as proof that an event is the latest/current one.
4. When multiple candidate events or results are present, compare event dates from titles/content and use the most recent completed/occurred event, not merely the freshest webpage.
5. Use the pipeline current time, when provided, to reason about whether a scheduled event may have already occurred. If the context contains a later scheduled/preview event for the target entity that is before the pipeline current time, but no final score/result for that later event, do not fall back to an older completed result as the "latest"; say the latest result cannot be confirmed from the evidence.
6. For sports scores/results, prefer evidence that explicitly says final score, result, completed game, box score, schedule result, or an official/statistical source page. Do not treat a single old-looking score as the most recent game unless the evidence proves it. Do not use a document with an older event date as support for the current latest event.
7. If answering with a score, price, count, date, or other numeric fact, briefly mention the supporting document title or source and include its [Doc n] citation.

The user question is:
"{question}"

The retrieved context is:
{context}

Please answer the question directly. Keep the answer concise, but attach [Doc n] citations to all evidence-backed claims.
"""

    no_rag_prompt_en = """You are answering the user's visual question.

You are about to be given the user's question and visual inputs. Answer directly based on the visual inputs and your own knowledge.

You should identify the user's requirements based on the question first. If the image or some images are completely unrelated to the question, please ignore them. Avoid forcibly associating unrelated images with the issue.

If the question asks for a latest, most recent, current, today/yesterday, recent factual value, final score, result, latest release, current price, current schedule, or similar time-sensitive fact, answer conservatively.

The user question is:
"{question}"
"""

    entity_candidates_prompt_en = """You are extracting likely entity candidates for a visual web RAG pipeline.

You are about to be given the user's question and one or more images.

The user question is:
"{question}"

## Task:
1. Infer the most likely visual entity or entities that are relevant to the question.
2. Focus on companies, products, teams, landmarks, artworks, books, songs, games, events, awards, or other named entities that would help web retrieval.
3. Prefer candidates that are most useful for answering the specific relation asked in the question. For example, if the question asks about an award, winning song, company, country, role, or collection, choose candidates that best anchor that relation in search.
4. If the image is only weakly relevant, you may rely more on the user question.
5. For each candidate, identify the most important missing slot that future retrieval should fill, such as award_name, winning_song, company_name, represented_country, product_line, game_title, or event_result.
6. Do not answer the user's question directly.

## Output requirements:
- Return JSON only.
- Use exactly this schema:
{{
  "candidates": [
    {{
      "name": "...",
      "type": "...",
      "aliases": ["..."],
      "confidence": 0.0,
      "reason": "...",
      "missing_slot": "..."
    }}
  ]
}}
- Return at most 3 candidates.
- "confidence" must be a number between 0 and 1.
- "aliases" may be empty.
- "missing_slot" should be a short snake_case label describing the most important fact still needed for that candidate.
- If candidate identity is uncertain, prefer a stable retrieval anchor over a flashy but weak guess.
- If you are highly uncertain, still return your best candidates rather than an empty list.
"""

    web_prompt_en = '''You are building a web search query for a visual RAG pipeline.

You are about to be given the user's question and one or more images.

The user question is:
"{question}"

You should construct the query for web retrieval based on the following requirements:

## Overall data format requirements:

    You should output only a single line search query.

## Content requirements:

    1. The query should be directly searchable by the browser. It should be concise, composed of keywords, not complete sentences, and avoid unnecessary punctuation.
    2. The information used in the query must be something you are certain of or clearly mentioned in the user question.
    3. Do not answer the question directly.

## Construction guidelines:

    ** 1. Understand the question:**
        Identify the user's requirements based on the question.

    ** 2. Understanding the image or images:**
        Identify the parts of the image that are easy to recognize and relevant or helpful to the user's question.
        
    ** 3. Construct query:**
        You must ensure that the information used to construct the query is trustworthy. Prioritize using reliable information such as events, awards and competitions to construct queries. If the image involves a person, do not assume their name or recognize them unless they are extremely famous (such as a national president).
'''

    entity_guided_web_prompt_en = '''You are building a web search query for a visual RAG pipeline.

You are about to be given the user's question, one or more images, and a small list of likely entity candidates extracted from the image.

The user question is:
"{question}"

Entity candidates:
{entity_candidates}

You should construct the query for web retrieval based on the following requirements:

## Overall data format requirements:

    You should output only a single line search query.

## Content requirements:

    1. The search results should be helpful in answering the question.
    2. Use the entity candidates as retrieval hints. Prefer the most plausible candidate, but if the top candidates are visually close, you may include 1-2 strong aliases or alternatives in the same query.
    3. The query should be concise, composed of keywords, not complete sentences, and avoid unnecessary punctuation.
    4. Do not answer the question directly.
    5. Do not add any explanations, quotes, numbering, or multiple options.

## Construction guidelines:

    ** 1. Choose the retrieval anchor: **
        Decide which candidate entity is most useful as the anchor for search. If the question is about a company, award or eventkeep the entity and the requested relation close together in the query.

    ** 2. Handle ambiguity carefully: **
        If image recognition is uncertain, include the strongest candidate plus one short alias or alternative name that would help search engines find the right entity. Do not include a long list of guesses.

    ** 3. Construct query: **
        Based on the information above, construct one search query.
        
The generated search query:'''

    freshness_prompt_en = """You are choosing a web-search freshness parameter for a visual RAG pipeline. This parameter controls the time range for the network search.

The user question is:
{question}

The generated search query is:
{query}

Current time:
{current_time}

Task:
Choose the freshness window that is most likely to retrieve the best evidence for answering the question.

Think about:
1. How quickly the target fact is likely to change.
2. Whether the user needs the newest available result or just a recent enough one.
3. Whether a narrow freshness filter might accidentally hide the best evidence.
4. Whether the question is about a completed event, an ongoing situation, a product release, a recent award, or a stable fact.

Guidance:
1. Return only one of: oneDay, oneWeek, oneMonth, oneYear, noLimit.
2. If the user explicitly gives a time range, follow it as closely as possible.
3. You should use a longer freshness that surely covers the events mentioned by users.
4. If the user explictly asks for events in days / weeks / months, use corresponding freshness. Otherwise, prefer longer fresnhess instead.
5. Do not overnarrow the freshness.
"""

    sufficiency_prompt_en = '''You are helping with a visual RAG QA progress.

You are about to be given the user's question, the user's visual input (image or images), a short list of entity candidates extracted from the image, and the retrieved context we have so far.

You need to judge whether the retrieved context is sufficient to answer the user the question.

User question:
{question}

Entity candidates:
{entity_candidates}

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

** 3. Identify the answer slot and the best retrieval anchor:**
    Determine the final fact slot that the question asks for, such as award_name, winning_song, company_name, represented_country, product_line, game_title, event_result, or role_title.
    Review the entity candidates and decide which candidate is the best retrieval anchor for filling that slot.
    If one candidate is clearly the most plausible, stay with that candidate unless the retrieved context strongly proves it is wrong.
    Do not drift to a different award, category, entity, or sub-question just because it appears in the current context.

** 4. Judge sufficiency by slot completion, not by topical overlap:**
    Return "YES" only when the current context is sufficient to fill the exact answer slot asked in the question for the correct target entity or event.

** 5. Construct addition:**
    If the context is insufficient, generate a single-line web browser query that targets the missing slot for the best candidate.
    The new query must NOT repeat or trivially paraphrase any previously tried query listed below:
    {previous_queries}
    
    If earlier queries already searched the same anchor + slot combination, change the query by adding a more specific target such as the exact category, winning work, official source, organization, or event-specific wording.

    Try to fetch the most important missing information with one targeted query.
'''

prompts = Prompts()
