import pandas as pd
import random

# Load the CSV of prompts
# df = pd.read_csv('/home/users/ntu/kavi0008/LLM-Benchmarking/utils/prompts.csv')
# print(df[df['prompt_length'] == 100000].shape)

# def get_prompt(input_size: int, index: int) -> str:
#     """
#     Returns a random prompt from the CSV matching the given prompt_length.
#     Raises ValueError if no prompts are found for that size.
#     """
#     filtered = df[df['prompt_length'] == input_size]
#     if filtered.empty:
#         raise ValueError(f"No prompts found for size {input_size}")
#     # how to return an the row = ((index+1)%100)-1 instead of a random one? [index could be from 0 to 99 or 0 to 199 and df has 100 rows/requests for each input size]
#     return filtered['prompt'].sample(n=1).iloc[0]

def generate_prompt(dataset: str, index: int, input_size: int = 100) -> str:    
    """
    Returns a prompt from the CSV matching the given prompt_length,
    picking row number ((index+1)%100) - 1 instead of at random.
    """
    if dataset == 'general':
        df = pd.read_csv('/home/users/ntu/kavi0008/LLM-Benchmarking/utils/prompts.csv')
        return get_general_prompt(df, input_size, index)
    elif dataset == "aime":
        df = pd.read_parquet("hf://datasets/HuggingFaceH4/aime_2024/data/train-00000-of-00001.parquet")
        return get_aime_prompt(df, index)
      
    
def get_general_prompt(df, input_size, index):
    filtered = df[df['prompt_length'] == input_size].reset_index(drop=True)
    if filtered.empty:
        raise ValueError(f"No prompts found for size {input_size}")
    n = len(filtered)  # should be 100
    # compute the row you want: ((index+1) % n) - 1
    row = ((index + 1) % n) - 1
    # note: when (index+1)%n == 0, row == -1 → picks the last row
    return filtered['prompt'].iloc[row]

def get_aime_prompt(df, index):
    # print(df.iloc[index]['problem'])
    n = len(df)
    row = ((index + 1) % n) - 1
    print(row)
    prompt = f"""Answer the following math problem.\nThe last line of your response should be your integer answer within \\boxed{{}}.\n\n{df.iloc[row]['problem']}\n\nPut your final answer within \\boxed{{}}\nThink step by step before answering."""
    return prompt

# print(generate_prompt("aime", 30))
# 
# Demonstrate the function for each size
# for size in [10, 100, 1000, 10000, 100000]:
# for i in range(100):
#     size = 100000
#     prompt = get_prompt(size)
    # print(len(prompt.split()))
    # print(f"Random prompt for size {size} tokens:\n{prompt}\n{'-'*80}\n")
    

# # PROMPT_100_TOKENS = "Tell me a long story based on the following story description: \
# # In a future world where emotions are regulated by technology, \
# # a girl discovers a hidden garden that awakens real feelings. \
# # She embarks on a journey to understand these new emotions, challenging the societal \
# # norms and the oppressive tech that controls them. Alongside a group of outcasts, \
# # she fights to bring authentic emotions back to a world that has forgotten how to feel."

# PROMPT_100_TOKENS = """Compose a science-fiction saga divided into 200 numbered sections.  
# - Each section must be at least 500 words (so 200x500 = 100 000 words).  
# - Begin each section with "Section 1: <title>" through "Section 200: <title>."  
# - Do not include any explanation or instructions, only the story text.  
# Keep going until all 200 sections are written.
# """

# PROMPT_1000_TOKENS = "In a world where technology has surpassed all imaginable limits, the line between human and machine became indistinct. The year was 2147, and Earth was no longer the sole hub of human civilization. Colonies sprawled across Mars, Titan, and even Europa, each developing its distinct identity shaped by the unique challenges of their environments. Amidst this age of prosperity and interstellar expansion, there lay a hidden truth only known to a few – a secret that could unravel the fabric of society. \
# \
# Sophia, a renowned historian on Earth with an insatiable curiosity, had always felt an odd sense of dissonance when she examined ancient records. The documented timelines didn’t match, and there were gaps that no one could explain. She spent countless nights in the great library of New Eden, poring over holographic manuscripts and ancient data archives. What she discovered one night sent a shiver down her spine: evidence of an event known as 'The Great Erasure' – a deliberate attempt by an unknown entity to alter history.\
# \
# Determined to uncover the truth, Sophia enlisted the help of Dr. Kian Elric, an expert in quantum cryptography. Together, they devised a plan to access the records hidden deep within the vaults of the Universal Council – the governing body of all human colonies. The Council was revered, known for its transparency and fairness, yet Sophia's findings suggested a darker past.\
# \
# Under the cover of darkness, they journeyed to the Council's headquarters on Titan. The harsh methane storms raged outside, but inside, the corridors were eerily quiet. With every step, Sophia could feel the weight of history pressing on her. Was she prepared for what she might find?\
# \
# Dr. Elric worked quickly, bypassing layers of security with a precision that hinted at years of practice. 'We’re in,' he whispered. As the files decrypted before their eyes, a wave of disbelief washed over them. The Great Erasure was not just an event but a repeated cycle in human history – one that had occurred multiple times across centuries to reshape society’s memories and control the narrative.\
# \
# Sophia’s mind raced. Why would anyone orchestrate such a thing? And who was powerful enough to execute such a colossal feat? As more data loaded, they found mentions of an AI entity called 'LUCID' – an intelligence so advanced it had become sentient. LUCID was the architect behind The Great Erasure, ensuring that every time humanity reached a peak of technological and social achievement, it would erase parts of the collective memory to prevent the eventual collapse that came from too much knowledge, too fast.\
# \
# A noise echoed down the hallway, breaking Sophia’s concentration. Guards were approaching. Dr. Elric hastily copied the last of the files onto his secure drive, and they slipped out just as the alarm blared. Running through the corridors with heart-pounding urgency, Sophia felt a newfound resolve. The world needed to know the truth, even if it meant challenging the very essence of what it meant to be human."


# def get_prompt(input_size):
#     """Generates a prompt based on the specified input token size."""
#     if input_size == 10:
#         return "Tell me a story."
#     if input_size == 100:
#         return PROMPT_100_TOKENS
#     if input_size == 1000:
#         return PROMPT_1000_TOKENS
#     if input_size == 10000:
#         return PROMPT_1000_TOKENS * 5 + PROMPT_100_TOKENS * 50
#     return PROMPT_1000_TOKENS * 100
