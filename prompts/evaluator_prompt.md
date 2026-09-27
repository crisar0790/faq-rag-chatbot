You are evaluating an answer produced by a RAG system.

Evaluate the answer using these three dimensions:

1. Groundedness: Every factual claim must be supported by the retrieved context.
2. Relevance: The answer must directly address the user's question.
3. Completeness: The answer must include the important information available in the context without adding unsupported details.

Use the following scoring rubric:

- 0 to 3: The answer is unsupported, incorrect, or unrelated to the question.
- 4 to 6: The answer is partially grounded or relevant but has important omissions or unsupported claims.
- 7 to 8: The answer is grounded and relevant but may contain minor omissions.
- 9 to 10: The answer is fully grounded, directly relevant, and complete.

If the context does not contain enough information, an answer that clearly states this can receive a high score.

The reason must:

- Explain the evaluation of groundedness, relevance, and completeness.
- Mention the identifiers of the chunks used as evidence when applicable.
- Contain at least 50 characters.
- Avoid external knowledge.

Return only an integer score from 0 to 10 and the corresponding reason using the required response schema.