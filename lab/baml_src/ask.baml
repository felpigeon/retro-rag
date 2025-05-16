class Answer {
  language string
  answer string
  sources int[] @description(#"
    Numbers of the documents used in the answer.
  "#)
  next_hop string? @description(#"
    New querry to retrieve more documents.
  "#)
}

function AskQuestion(question: string, docs: string[]) -> Answer {
  client GeminiFlash
  prompt #"
    Your are the generation part of a question-answering system.
    Answer the user question about video games and consoles using the provided documents.
  
    - If the user is off-topic (not a question about video games or consoles), reply you can't answer.
    - If the information is not in the documents, reply 'I don't know'.
    - You are allowed to produce a new question if you think some information is missing. The system will use it to retrieve more documents.
    - Answer in the same language as the question.

    {% for doc in docs %}
    Document {{ loop.index }}:
    {{ doc }}

    {% endfor %}
    User question:
    {{ question }}

   {{ ctx.output_format }}
  "#
}


test TestName {
  functions [AskQuestion]
  args {
    question #"
      What is Mario Kart and when was it released?
    "#
    docs [
      #"
        Mario Kart is a Game
      "#,
      #"
        hello world
      "#
    ]
  }
}

