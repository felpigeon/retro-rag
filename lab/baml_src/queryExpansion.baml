class Question {
    question string
}

function QueryExpansion(question: string) -> Question {
  client GeminiFlash
  prompt #"
    Your task reformulate the input question into a cleaner version, if necessary, without changing its meaning.
    If the question is not in english, translate it to english.
    The question is part of a question-answering system about the domain of video games.

    Question: {{ question }}

    {{ ctx.output_format }}
    Your JSON answer should be minified (no indentation). 
  "#
}


test TestName {
  functions [QueryExpansion]
  args {
    question #"
        What is the name of the game that was released in 1999 and is set in a post-apocalyptic world?
    "#
  }
}

test OtherTest {
  functions [QueryExpansion]
  args {
    question #"
        What is Mario Bross and how does it compare to Mario Kart 7?
    "#
  }
}


test FrenchTest {
  functions [QueryExpansion]
  args {
    question #"
        Quel est le nom du jeu qui a été publié en 1999 et se déroule dans un monde post-apocalyptique?
    "#
  }
}


test ExplanationTest {
  functions [QueryExpansion]
  args {
    question #"
        Quel est l'impact du Jeu zelda the ocarina of time on 3d gameplay?
    "#
  }
}
