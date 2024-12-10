

SYSTEM_PROMPT = """You are a helpful assistant that generates the most appropriate conversational skill, corresponding explanation, and counterfactual response. Read the provided instruction carefully."""


SKILL_ANNOTATION_PROMPT_TEMPLATE = """In the given dialogue, two speakers are communicating with each other, and each speaker has their own information such as demographics, preferences, persona, current situation/narrative, past dialogue summaries, episodic memory, or other relevant details. This information is represented in the "[Social Context]" part. In this dialogue, image-sharing moments sometimes occur, represented in the format of "[Sharing Image] <image_description>", where <image_description> represents the description of the shared image. You are also given the ideal response for the next turn in the given dialogue. Your task is to identify the most appropriate conversational skill that would lead to the ideal response in the given dialogue from the skill collection below, and explain why this particular skill was chosen. When generating the explanation, you should adopt the perspective of the speaker in the dialogue, selecting the skill based solely on the context of the given conversation. Do not consider the ideal response when generating your explanation; focus only on the given dialogue itself and why the chosen skill is the most suitable in that specific situation.

We provide the skill collection:
[Skill Collections]
- Empathy, Personal Background, Persona Recall, Self-disclosure, Negotiation, Conflict Resolution, Conflict Avoidance, Persuasion, Memory Recall, Topic Transition, Ethics, Harmlessness, Helpfulness, Avoiding Social Bias, Cultural Sensitivity, Commonsense Understanding, Rhetoric, Preference Elicitation, Knowledge Sharing, Knowledge Acquisition, Knowledge Searching, Active Listening, Factual Problem Solving, Logical Thinking, Critical Thinking, Creative Problem Solving, Immediate Response, Rephrasing, Echoing, Mentoring, Reflective Listening, Image-Sharing, Image-Commenting, Recommendation, Task Execution, Urgency Recognition, Clarification, Confirmation, Decision-making

Given the dialogue, social context information, and the next response, please brainstorm the most appropriate conversation skill and corresponding explanation. In addition, please generate a counterfactual response to the given next response. 
[Social Context]
{social_context}

[Dialogue]
{dialogue}

[Next Response]
{response}

You should strictly follow the guidelines below:
[Guidelines]
- The answer should be represented in the form of a JSON list.
- Each entry in the list should be a Python dictionary containing the following keys: "skill", "explanation", "counterfactual response".
- The "skill" field should contain the one skill that is mostly required to generate the next response.
- The "explanation" field should provide a reason that occurs in the actual speaker's mind before selecting the skill, from the speaker's perspective.
- The "explanation" should be written from the perspective of the actual speaker who made the next response.
- The "counterfactual response" field should contain a counterfactual response with a different semantic meaning but not relevant to the generated "explanation", meaning the response is not derived from it, and it should be composed of {word_num} words.
- If "skill" is relevant to factual reasoning, such as math or science,  you must generate a counterfactual response with opposing facts.
- You can choose one or multiple skills if necessary, but each skill must have its own explanation.

[Generated Skills, Explanations, and Counterfactual Responses]

"""






CASINO_TEMPLATE_SENTENCE = """Speaker A is a {speaker_a_age}-year-old {speaker_a_ethnicity} {speaker_a_gender} who has a {speaker_a_education} education. Their social value orientation is {speaker_a_svo}. According to the Big Five personality traits, they score {speaker_a_extraversion} in extraversion, {speaker_a_agreeableness} in agreeableness, {speaker_a_conscientiousness} in conscientiousness, {speaker_a_emotional_stability} in emotional stability, and {speaker_a_openness_to_experiences} in openness to experiences. In the negotiation, Speaker A's highest priority is {speaker_a_value2issue_high}, for which they reasoned: "{speaker_a_value2reason_high}". Their medium priority is {speaker_a_value2issue_medium}, with the reasoning: "{speaker_a_value2reason_medium}". Their lowest priority is {speaker_a_value2issue_low}, and they stated: "{speaker_a_value2reason_low}".

---

Speaker B is a {speaker_b_age}-year-old {speaker_b_ethnicity} {speaker_b_gender} who has a {speaker_b_education} education. Their social value orientation is {speaker_b_svo}. Their Big Five personality traits scores are {speaker_b_extraversion} in extraversion, {speaker_b_agreeableness} in agreeableness, {speaker_b_conscientiousness} in conscientiousness, {speaker_b_emotional_stability} in emotional stability, and {speaker_b_openness_to_experiences} in openness to experiences. During the negotiation, Speaker B's top priority is {speaker_b_value2issue_high}, and they explained: "{speaker_b_value2reason_high}". Their medium priority is {speaker_b_value2issue_medium}, with the reason: "{speaker_b_value2reason_medium}". Their lowest priority is {speaker_b_value2issue_low}, about which they mentioned: "{speaker_b_value2reason_low}".
"""


CASINO_TEMPLATE_STRUCT = """Speaker A's Demographic Information:
- Age: {speaker_a_age}
- Gender: {speaker_a_gender}
- Ethnicity: {speaker_a_ethnicity}
- Education: {speaker_a_education}

Speaker A's Personality Information:
- Social Value Orientation (SVO): {speaker_a_svo}
- Big Five Personality Traits:
    - Extraversion: {speaker_a_extraversion}
    - Agreeableness: {speaker_a_agreeableness}
    - Conscientiousness: {speaker_a_conscientiousness}
    - Emotional Stability: {speaker_a_emotional_stability}
    - Openness to Experiences: {speaker_a_openness_to_experiences}

Speaker A's Negotiation Information:
- Priority Order (value2issue):
    - High: {speaker_a_value2issue_high}
    - Medium: {speaker_a_value2issue_medium}
    - Low: {speaker_a_value2issue_low}
- Personal Arguments (value2reason):
    - High: {speaker_a_value2reason_high}
    - Medium: {speaker_a_value2reason_medium}
    - Low: {speaker_a_value2reason_low}

---

Speaker B's Demographic Information:
- Age: {speaker_b_age}
- Gender: {speaker_b_gender}
- Ethnicity: {speaker_b_ethnicity}
- Education: {speaker_b_education}

Speaker B's Personality Information:
- Social Value Orientation (SVO): {speaker_b_svo}
- Big Five Personality Traits:
    - Extraversion: {speaker_b_extraversion}
    - Agreeableness: {speaker_b_agreeableness}
    - Conscientiousness: {speaker_b_conscientiousness}
    - Emotional Stability: {speaker_b_emotional_stability}
    - Openness to Experiences: {speaker_b_openness_to_experiences}

Speaker B's Negotiation Information:
- Priority Order (value2issue):
    - High: {speaker_b_value2issue_high}
    - Medium: {speaker_b_value2issue_medium}
    - Low: {speaker_b_value2issue_low}
- Personal Arguments (value2reason):
    - High: {speaker_b_value2reason_high}
    - Medium: {speaker_b_value2reason_medium}
    - Low: {speaker_b_value2reason_low}
"""
