from llama_index.prompts import PromptTemplate
from utils import *

from llama_index.llms import OpenAI
from llama_index.prompts import PromptTemplate
import ast

llm = OpenAI(temperature=0.1, model="gpt-4") # Better results, but very slow and expensive
# llm = OpenAI(temperature=0.1, model="gpt-3.5-turbo")

# Create a ServiceContext with the LLM
service_context = ServiceContext.from_defaults(llm=llm)


expert_retrieve_public_template = (
    "{title}\n"
    "{question}\n"
    "Use the following resources: {use}\n"
)

expert_retrieve_private_template = (
    "{title}\n"
    "{question}\n"
    "Use the following resources, as well as corresponding solutions or answer keys: {use}\n"
)

expert_answer_template = (
    "Given the following context, answer the student's question to the best of your ability.\n"
    "---------------------\n"
    "{context_str}"
    "---------------------\n"
    "Question Title: {title}\n"
    "Question: {question}\n"
    "---------------------\n"
)

to_student_template = (
    "Public context:\n"
    "---------------------\n"
    "{public_context}"
    "\n---------------------\n"
    "Private context:\n"
    "---------------------\n"
    "{private_context}"
    "\n---------------------\n"
    "Correct Answer:\n"
    "---------------------\n"
    "{answer}"
    "\n---------------------\n"
    "You are a helpful agent giving students a hint without revealing the answer.\n"
    "Given the correct answer and potentially relevant context, output the 0-index of the most useful public context, and a link to a helpful resource.\n"
    "Example Output: (2, 'https://en.wikipedia.org/wiki/Newton%27s_laws_of_motion')\n"
    "Example Output 2: (0, 'https://math.stackexchange.com/questions/265917/intuitive-explanation-of-a-definition-of-the-fisher-information')\n"
    "Output:"
)


student_answer_template = (
    "Given the following context, answer the student's question to the best of your ability.\n"
    "---------------------\n"
    "{context_str}"
    "---------------------\n"
    "Potentially helpful resource:"
    "{link}"
    "---------------------\n"
    "Question Title: {title}\n"
    "Question: {question}\n"
    "---------------------\n"
)


student_vanilla_answer_template = (
    "Given the following context, answer the student's question to the best of your ability.\n"
    "---------------------\n"
    "{context_str}"
    "---------------------\n"
    "Question Title: {title}\n"
    "Question: {question}\n"
    "---------------------\n"
)



def nodes_to_str(retrieved_nodes):
    return "\n---------------\n".join([f"Context {index}: \n\n" + r.get_content() for index, r in enumerate(retrieved_nodes)])


def run_model(query_dict, public_index, private_index, llm):
    """
    query_dict is a dict with the following format:
    {
        "title": "PSET 4 Question 3",
        "question": "I would expect $Y_i$ here to follow a Bernoulli distribution since it can only take on two values 0 or 1. However, I am confused on how to define the Bernoulli parameter for $Y_i$. We are given $T_i$ comes from $exp(1)$. I was thinking that $P(Y_i=1)=\\int_{0}^{\\infty}(1-F(t))exp(-t)dt$. Since we would sweep through all possible values of T if $P(Y_i=1)=P(X_i>T_i)=P(X_i>t|T_i=t)\\cdot exp(-t)$. Am I on the right track? Since I don't know $F(t)$ for this problem, I'm not sure if I can leave the parameter definition with the integral in it.",
        "ta_response": "Yes, you're on the right track! The integral can be further simplified - you'll get an expectation of a function of $X_i$.",
        "use": "Problem Set 4, Lecture 28 Causal Inference, Recitation 3, Review sheet 2"
    },
    """
    question = query_dict['question']
    title = query_dict['title']

    private_nodes = test_retrieval(private_index, expert_retrieve_private_template.format(title=title, question=question, use=query_dict['use']))
    public_nodes = test_retrieval(public_index, expert_retrieve_public_template.format(title=title, question=question, use=query_dict['use']))

    print("Private context nodes for the expert:")
    print(nodes_to_str(private_nodes))
    print("Public context nodes for the expert:")
    print(nodes_to_str(public_nodes))

    expert_answer_prompt = expert_answer_template.format(context_str=nodes_to_str(private_nodes + public_nodes), 
                                                         title=title, question=question)
    expert_answer = llm.complete(expert_answer_prompt)
    print("Expert answer: ", expert_answer)

    to_student_prompt = to_student_template.format(public_context=nodes_to_str(public_nodes), 
                                                   private_context=nodes_to_str(private_nodes), 
                                                   answer=expert_answer)
    
    to_student = llm.complete(to_student_prompt)
    for _ in range(3):
        try:
            best_idx, link = eval(str(to_student))
            assert(best_idx < len(public_nodes))
            break
        except:
            to_student = llm.complete(to_student_prompt)
    
    print("To student: ", to_student)
    best_idx, link = eval(str(to_student))
    print(best_idx, link)

    extra_nodes = test_retrieval(public_index, str(expert_answer))
#    print("extra nodes: ", extra_nodes)

    student_answer_prompt = student_answer_template.format(context_str=nodes_to_str([public_nodes[best_idx]] + extra_nodes), 
                                                           link=link, title=title, question=question)
    student_answer = llm.complete(student_answer_prompt) 

    print("Student answer: ", student_answer)
    return expert_answer, student_answer


questions = fetch_question_list()

public_index = index_from_dir('./data/public_persist')
private_index = index_from_dir('./data/private_persist')

for q in questions[:1]:
    print(q)
    run_model(q, public_index, private_index, llm)