from llama_index.prompts import PromptTemplate
from utils import *
import random
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
    "Given the correct answer and potentially relevant context, output the 0-index of the most useful public context, and a link to a helpful resource that isn't already directly mentioned in the public context.\n"
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


def raw_output(query_dict, public_index, llm):
    title=query_dict['title']
    question=query_dict['question']
    use=query_dict['use']
    public_nodes = test_retrieval(public_index, expert_retrieve_public_template.format(title=title, question=question, use=use))

    raw_answer_prompt = student_vanilla_answer_template.format(context_str=nodes_to_str(public_nodes), 
                                                           title=title, question=question)
    raw_answer = llm.complete(raw_answer_prompt) 

    return raw_answer 


def run_model(query_dict, public_index, private_index, llm, debug=False):
    """
    query_dict is a dict with the following format:
    {
        "title": 
        "question": 
        "ta_response": 
        "use": 
    },
    """
    question = query_dict['question']
    title = query_dict['title']

    private_nodes = test_retrieval(private_index, expert_retrieve_private_template.format(title=title, question=question, use=query_dict['use']))
    public_nodes = test_retrieval(public_index, expert_retrieve_public_template.format(title=title, question=question, use=query_dict['use']))

    if debug:
        print("Private context nodes for the expert:")
        print(nodes_to_str(private_nodes))
        print("Public context nodes for the expert:")
        print(nodes_to_str(public_nodes))

    expert_answer_prompt = expert_answer_template.format(context_str=nodes_to_str(private_nodes + public_nodes), 
                                                         title=title, question=question)
    expert_answer = llm.complete(expert_answer_prompt)
    if debug:
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
    
    if debug:
        print("To student: ", to_student)
        print(best_idx, link)

    extra_nodes = test_retrieval(public_index, str(expert_answer))
    if debug:
        print("Extra nodes for the student: ===========\n")
        for node in extra_nodes:
            print(node.get_content())

    student_answer_prompt = student_answer_template.format(context_str=nodes_to_str([public_nodes[best_idx]] + extra_nodes), 
                                                           link=link, title=title, question=question)
    student_answer = llm.complete(student_answer_prompt) 

    if debug:
        print("Student answer: ", student_answer)
    return expert_answer, student_answer



public_index = index_from_dir('./data/public_persist')
private_index = index_from_dir('./data/private_persist')

def test():
    questions = fetch_question_list("1432")
    q = questions[0]
    ta = q['ta_response']
    ex, stu = run_model(q, public_index, private_index, llm, debug=True)
    pub = raw_output(q, public_index, llm)
    raw = llm.complete(q['question'])

    print("Expert answer: ", ex)
    print("Student answer: ", stu)
    print("TA answer: ", ta)
    print("Public context answer: ", pub)
    print("Direct LLM: ", llm.complete(q['question']))

test()

def run_test(test_class="1432"):
    questions = fetch_question_list(test_class)
    scores = [0,0,0]
    random.shuffle(questions)

    for q_idx, q in enumerate(questions):
        print("Title: ", q['title'])
        print("Question: \n=============\n: ", q['question'])
        print("=============\nPlease wait... generating outputs. This might take a few seconds")
        ta = q['ta_response']
        ex, stu = run_model(q, public_index, private_index, llm)
        pub = raw_output(q, public_index, llm)
        raw = llm.complete(q['question'])

        outputs = [(stu,0), (ta,1), (raw,2)]
        random.shuffle(outputs)
        
        for idx, output in enumerate(outputs):
            print(f"Output {idx}: ===========")
            print(output[0])
            print()
        
        for idx in range(3):
            score = input(f"Score the output for output {idx}: (1-5)")
            scores[outputs[idx][1]] += int(score)

        print("Model (Student) | TA | Naive LLM")
        print([s/(1+q_idx) for s in scores])

