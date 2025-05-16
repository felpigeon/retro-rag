function ExtractEntities(question: string) -> Entity[] {
  client GeminiFlash
  prompt #"
    Extract all the relevant entities from the user question on the video games and consonles domain.
    - Resolve entities to their canonic form (example: LOTR -> Lord of the Rings).

    User question:
    {{ question }}

   {{ ctx.output_format }}
  "#
}


test ExtractEntitiesTest {
  functions [ExtractEntities]
  args {
    question #"
      What is Mario Kart 
    "#
  }
}