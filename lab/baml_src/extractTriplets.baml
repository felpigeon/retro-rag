function ExtractTriples(question: string, documents: string) -> string[][] {
  client GeminiFlash
  prompt #"
    Your task is to find facts that help answer an input question.
    You should present these facts as knowlege triples, which are structured as ("subject", "predicate", "object").
    Example:
    Question: When was Neville A. Stanton’s employer founded?
    Facts: ("Neville A. Stanton", "employer", "University of Southampton"), ("University of Southampton", "founded in",
    "1862")
    Now you are given some documents:
    {{ documents}}

    Based on these documents and some preliminary facts provided below,
    find additional supporting fact(s) that may help answer the following question.
    Resolve the pronouns and other references in the documents to the entities they refer to. 
    Note: if the information you are given is insufficient, output only the relevant facts you can find.

    Question: {{ question }}

    {{ ctx.output_format }}
    Your JSON answer should be minified (no indentation).
  "#
}
