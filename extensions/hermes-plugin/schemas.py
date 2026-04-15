"""Tool schemas — what the LLM sees for manual recall/retain/reflect/stats."""

CORTILOOP_RECALL = {
    "name": "cortiloop_recall",
    "description": (
        "Search long-term memory for relevant context. Uses multi-probe retrieval: "
        "semantic similarity, keyword matching, graph traversal, and temporal filtering. "
        "Use this when you need to recall past conversations, user preferences, or "
        "previously learned information beyond what was auto-injected."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The query to search memories for",
            },
            "top_k": {
                "type": "integer",
                "description": "Maximum number of results (default 5)",
                "default": 5,
            },
        },
        "required": ["query"],
    },
}

CORTILOOP_RETAIN = {
    "name": "cortiloop_retain",
    "description": (
        "Explicitly store important information into long-term memory. "
        "Use this for critical facts, corrections, or insights that the auto-retain "
        "might miss or that you want to store with specific task context. "
        "The attention gate filters noise automatically."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "The text content to remember",
            },
            "task_context": {
                "type": "string",
                "description": "Current task description for relevance scoring",
                "default": "",
            },
        },
        "required": ["text"],
    },
}

CORTILOOP_REFLECT = {
    "name": "cortiloop_reflect",
    "description": (
        "Trigger a deep consolidation cycle on stored memories. "
        "Detects procedural patterns, generates mental models, runs decay sweep, "
        "and prunes duplicates. Use after long sessions or when memory feels fragmented."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
    },
}

CORTILOOP_STATS = {
    "name": "cortiloop_stats",
    "description": "Get CortiLoop memory system statistics — unit count, observation count, storage usage.",
    "parameters": {
        "type": "object",
        "properties": {},
    },
}
