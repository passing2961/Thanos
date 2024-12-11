SODA_CHATGPT_TEMPLATE = """You will be generating the next turn of a given dialogue between two people. Your response should usually be 1-2 sentences. Alongside the dialogue (which is provided line-by-line, where a new-line means the speaker changed), you’ll be given some context about the two participants of the dialogue, e.g., their relationship, situation, etc.

context:
[narrative]
dialogue:
[dialogue]
What is the most appropriate next utterance (3 sentences max)?"""

DOCTOR_CHATGPT_TEMPLATE = """Generate the most plausible next response considering the dialogue history. You can refer to the rationale, but you should ignore the rationale if it misleads the next response. Do not try to put too much information in the next response. You should follow the style of the history.

[Rationale]
{rationale}

[Social Context]
{social_context}

[Dialogue]
{dialogue}

[Next Response]
{next_speaker}
"""


THANOS_NEXT_RESP_TEMPLATE_BOTH = """Your task is to generate the most appropriate next response of a given dialogue between two speakers. You'll be given some social context about the two speakers of the dialogue, e.g., their relationship, demographic, preference, persona, or situation, etc. Additionally, you will receive an explanation that interprets the current dialogue and suggests strategic skills for the next response. You may refer to this explanation, but should ignore it if it misleads the response. Do not include too much information in the next response, and be sure to follow the style of the dialogue.

[Social Context]
{social_context}

[Dialogue]
{dialogue}

[Explanation and Skill]
{explanation} Thus, the most appropriate conversational skill for the next response is {skill}.

[Next Response]
{next_speaker}
"""

THANOS_NEXT_RESP_TEMPLATE_SKILL = """Your task is to generate the most appropriate next response of a given dialogue between two speakers. You'll be given some social context about the two speakers of the dialogue, e.g., their relationship, demographic, preference, persona, or situation, etc. Additionally, you will receive an explanation that interprets the current dialogue and suggests strategic skills for the next response. Please be sure to follow the style of the dialogue.

[Social Context]
{social_context}

[Dialogue]
{dialogue}

[Conversational Skill]
The most appropriate conversational skill for the next response is {skill}.

[Next Response]
{next_speaker}
"""

THANOS_NEXT_RESP_TEMPLATE_RATIONALE = """Your task is to generate the most appropriate next response of a given dialogue between two speakers. You'll be given some social context about the two speakers of the dialogue, e.g., their relationship, demographic, preference, persona, or situation, etc. Additionally, you will receive an explanation that interprets the current dialogue. You may refer to this explanation, but should ignore it if it misleads the response. Do not include too much information in the next response, and be sure to follow the style of the dialogue.

[Social Context]
{social_context}

[Dialogue]
{dialogue}

[Explanation]
{explanation}

[Next Response]
{next_speaker}
"""

THANOS_NEXT_RESP_SYSTEM_MESSAGE = "You are a helpful assistant."


OUR_BASE_NEXT_RESP_TEMPLATE = """Your task is to generate the most appropriate next response of a given dialogue between two speakers. You'll be given some social context about the two speakers of the dialogue, e.g., their relationship, demographic, preference, persona, or situation, etc.

[Social Context]
{social_context}

[Dialogue]
{dialogue}

[Next Response]
{next_speaker}
"""

OUR_BASE_SKILL_TEMPLATE = """In the given dialogue, two speakers are communicating with each other, and each speaker has their own information such as demographics, preferences, persona, current situation/narrative, past dialogue summaries, episodic memory, or other relevant details. This information is represented in the "[Social Context]" part. In this dialogue, image-sharing moments sometimes occur, represented in the format of "[Sharing Image] <image_description>", where <image_description> represents the description of the shared image. Your task is to identify the most appropriate conversational skill for the next response of the given dialogue from the skill collection below, and explain why this particular skill was chosen. When generating the explanation, you should adopt the perspective of the speaker in the dialogue, selecting the skill based solely on the context of the given conversation. The output format should be as follows: "Explanation: (write an explanation for why the chosen skill is selected.) [RESULT SKILL] (A conversational skill that fits the situation.)"

We provide the skill collection:
[Skill Collections]
- Empathy, Personal Background, Persona Recall, Self-disclosure, Negotiation, Conflict Resolution, Conflict Avoidance, Persuasion, Memory Recall, Topic Transition, Ethics, Harmlessness, Helpfulness, Avoiding Social Bias, Cultural Sensitivity, Commonsense Understanding, Rhetoric, Preference Elicitation, Knowledge Sharing, Knowledge Acquisition, Knowledge Searching, Active Listening, Factual Problem Solving, Logical Thinking, Critical Thinking, Creative Problem Solving, Immediate Response, Rephrasing, Echoing, Mentoring, Reflective Listening, Image-Sharing, Image-Commenting, Recommendation, Task Execution, Urgency Recognition, Clarification, Confirmation, Decision-making

[Social Context]
{social_context}

[Dialogue]
{dialogue}

Explanation:
"""

OUR_BASE_SKILL_SYSTEM_MESSAGE = """You are an excellent skill predictor that generates the most appropriate conversational skill for the next turn response in the given dialogue and social context. Before generating the skill, please think about which skill is appropriate and then generate the skill."""
OUR_BASE_NEXT_RESP_SYSTEM_MESSAGE = "You are a helpful assistant."

# You MUST select the most appropriate skill from the collection of conversational skills shown below.
# ### Skill Collections:
# {skill_collection}

THANOS_SKILL_TEMPLATE = """### Task Description:
A dialogue and social context containing the speaker's demographics, preferences, persona, current situation/narrative, past dialogue summaries, episodic memory, or other relevant details are provided. During this dialogue, image-sharing moments may occur, represented by the format "[Sharing Image] <image_description>", where "<image_description>" represents the description of the shared image. Your task is to imagine yourself as the actual speaker who needs to respond in the next conversational turn. You will first generate the internal thought process behind selecting the appropriate conversational skill, and then generate the most appropriate conversational skill itself. The output format should be as follows: "### Explanation: (write an explanation for why the chosen skill is selected.) [RESULT SKILL] (A conversational skill that fits the situation.)"

### Social Context:
{social_context}

### Dialogue:
{dialogue}

### Explanation: """


THANOS_SKILL_SYSTEM_MESSAGE = """You are an excellent skill predictor that generates the most appropriate conversational skill for the next turn response in the given dialogue and social context. Before generating the skill, please think about which skill is appropriate and then generate the skill."""
