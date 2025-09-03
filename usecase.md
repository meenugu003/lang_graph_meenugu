our Use Case: Unified Product Chatbot with Decision Trees

Problem

You have ~100 products under the same line of business.

Each product has its own documentation (Markdown files).

It’s hard for users to read all docs and figure out how to onboard or use the right product.

Goal

Build a chatbot that provides a unified interface to guide users.

Instead of dumping docs, the bot navigates the user through a decision tree and finally suggests a strategic pattern (solution) for the chosen product.

Approach

Each product team uploads Markdown files.

An LLM processes these docs to generate:

Decision Trees (questions, branching logic, strategies)

Strategic Patterns (recommended solution paths)

Decision trees are converted into LangGraph StateGraph objects.

Store the compiled trees in Postgres (source of truth) and cache them in LangMem for fast chatbot execution.

Chatbot Flow

User asks about a product → chatbot fetches the corresponding decision tree (from LangMem or Postgres).

The chatbot executes the decision tree as a LangGraph:

Each node = a question

Each edge = a user choice → next step

Each leaf = a strategy pattern / solution

User progresses step by step → reaches the final strategy recommendation.

LangMem also tracks user’s current node, so they can resume mid-way.

Tech Stack

Postgres → Stores decision trees, nodes, edges, versions.

LangGraph → Represents and executes decision trees (StateGraph).

LangMem → Caches compiled trees + stores user session progress.

LLM → Generates trees/strategic patterns from docs.
