import yaml
from typing import List, Literal, Optional, Dict, Any, TypedDict
from pydantic import BaseModel, Field, ValidationError

class Option(BaseModel):
    label: str
    next: str

class ProvenanceItem(BaseModel):
    file: str
    score: float

class Provenance(BaseModel):
    __root__: Dict[str, List[ProvenanceItem]]

class Node(BaseModel):
    id: str
    kind: Literal["question", "leaf"]
    text: Optional[str] = None
    options: Optional[List[Option]] = None
    emitsPattern: Optional[str] = None

class DecisionTreeSpec(BaseModel):
    start: str
    nodes: List[Node]
    provenance: Optional[Provenance] = None

class DecisionTree(BaseModel):
    apiVersion: str
    kind: str
    metadata: Dict[str, Any]
    spec: DecisionTreeSpec


from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Optional, Literal
from langchain_core.messages import HumanMessage

# Define the state for our graph
class TreeState(TypedDict):
    """Represents the state of our graph.
    `current_node_id`: The ID of the current node the user is at.
    `user_choice`: The choice the user just made.
    `output`: The final recommendation or solution.
    """
    current_node_id: str
    user_choice: Optional[str] = None
    output: Optional[str] = None

def build_langgraph_from_tree(tree_obj: DecisionTree):
    """
    Builds a LangGraph StateGraph from a Pydantic DecisionTree object.
    """
    workflow = StateGraph(TreeState)
    nodes_by_id = {node.id: node for node in tree_obj.spec.nodes}

    # Define a function to be used as a LangGraph node for questions
    def question_node_executor(state: TreeState):
        node_id = state['current_node_id']
        node_data = nodes_by_id[node_id]
        
        # This function presents the question and options to the user
        print(f"\n[Bot]: {node_data.text}")
        for i, option in enumerate(node_data.options):
            print(f"  {i+1}. {option.label}")
        
        # We need to block here and wait for the user's input,
        # which will be passed to the next state update.
        # This is the "user-facing" part of the node.
        return {"current_node_id": node_id}

    # Define a function for the final leaf nodes
    def leaf_node_executor(state: TreeState):
        node_id = state['current_node_id']
        node_data = nodes_by_id[node_id]
        
        # The final recommendation
        output_text = f"\n[Bot]: Your recommended strategic pattern is: **{node_data.emitsPattern}**"
        print(output_text)
        return {"output": output_text}

    # Define the conditional edge logic based on user choice
    def router(state: TreeState):
        current_node_id = state['current_node_id']
        user_choice = state.get('user_choice')
        
        if not user_choice:
            # If no choice is made, stay at the current node (e.g., re-prompt)
            return current_node_id
        
        current_node_data = nodes_by_id[current_node_id]
        
        for option in current_node_data.options:
            if option.label.lower() == user_choice.lower() or str(current_node_data.options.index(option) + 1) == user_choice:
                return option.next
        
        # Fallback for invalid input
        return current_node_id

    # Add nodes and edges to the workflow
    for node in tree_obj.spec.nodes:
        if node.kind == "question":
            workflow.add_node(node.id, question_node_executor)
        elif node.kind == "leaf":
            workflow.add_node(node.id, leaf_node_executor)

    # Add edges
    for node in tree_obj.spec.nodes:
        if node.kind == "question":
            workflow.add_conditional_edges(
                node.id,
                router,
                {opt.next: opt.next for opt in node.options} # Maps user choices to next nodes
            )
        elif node.kind == "leaf":
            workflow.add_edge(node.id, END)

    # Set the starting point and return compiled graph
    workflow.set_entry_point(tree_obj.spec.start)
    return workflow.compile()





# The YAML data from your prompt
yaml_data = """
apiVersion: lob.ai/v1
kind: DecisionTree
metadata:
  treeId: 7a0f...
  product: stream-proc
  version: 2025.09.02-1
  sourceBundle: bndl-123
spec:
  start: n-root
  nodes:
    - id: n-root
      kind: question
      text: "What ingestion pattern fits your workload?"
      options:
        - label: "Exactly-once streaming"
          next: n-eos
        - label: "At-least-once batch"
          next: n-batch
    - id: n-eos
      kind: question
      text: "Throughput target?"
      options:
        - label: ">= 50k events/sec"
          next: n-high
        - label: "< 50k events/sec"
          next: leaf-s1
    - id: leaf-s1
      kind: leaf
      emitsPattern: pat-EO-50k
    - id: n-batch
      kind: question
      text: "Batch size?"
      options:
        - label: "Large"
          next: leaf-s2
        - label: "Small"
          next: leaf-s3
    - id: leaf-s2
      kind: leaf
      emitsPattern: pat-AL-large
    - id: leaf-s3
      kind: leaf
      emitsPattern: pat-AL-small
    - id: n-high
      kind: question
      text: "Data source?"
      options:
        - label: "External services"
          next: leaf-s4
        - label: "Internal systems"
          next: leaf-s5
    - id: leaf-s4
      kind: leaf
      emitsPattern: pat-EO-high-ext
    - id: leaf-s5
      kind: leaf
      emitsPattern: pat-EO-high-int
  provenance:
    n-eos:
      - file: md/ingestion.md#exactly-once
        score: 0.83
"""

if __name__ == "__main__":
    try:
        data = yaml.safe_load(yaml_data)
        tree = DecisionTree.parse_obj(data)
        
        # Build the LangGraph
        app = build_langgraph_from_tree(tree)
        
        # Initial state
        state = {"current_node_id": tree.spec.start, "user_choice": None, "output": None}
        
        print("Welcome to the Product Decision Chatbot!")
        print("Type 'exit' to end the conversation.")
        
        # Main chat loop
        while True:
            # First, invoke the graph to get the question for the current node
            # The .invoke() method executes the current node's function
            # which in our case, prints the question.
            result = app.invoke(state, config={"recursion_limit": 50})

            # Check for the final output (leaf node)
            if result.get("output"):
                print(result["output"])
                break
                
            # Get user input for the next step
            user_input = input("\n[You]: ")
            if user_input.lower() == 'exit':
                print("Thank you for using the chatbot! Goodbye.")
                break
            
            # Update the state with the user's choice and re-invoke the graph
            state['user_choice'] = user_input
            
            # The graph will now route to the next node based on the user's choice
            # and repeat the loop.
    except ValidationError as e:
        print("Error parsing YAML:", e.errors())
    except Exception as e:
        print(f"An error occurred: {e}")
