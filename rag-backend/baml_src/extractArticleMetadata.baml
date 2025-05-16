enum EntityType {
  Game
  Console
  Publisher
}

class Entity {
  name string
  type EntityType
}


class ArticleMetadata {
  entities Entity[]
  triples string[][]
}

function ExtractArticleMetadata(article: string) -> ArticleMetadata {
  client GeminiFlash
  prompt #"
    # Instruction
    Your task is to construct an RDF (Resource Description Framework) graph for the video games and consoles domain from the given article and named entity lists.
    Respond with a JSON list of triples, with each triple representing a relationship in the RDF graph.
    Pay attention to the following requirements:
    - Triple should ideally contain at least one, preferably two, of the named entities in the list for each passage, but you can also extract any relevant triple related to video games domain (genre, gameplay, critic, etc.).
    - Clearly resolve pronouns to their specific names to maintain clarity.
    - Add only entities that are clearly of one of the given type and resolve their name to their canonic form (example: LOTR -> Lord of the Rings).

    {{ ctx.output_format }}
    Your JSON answer should be minified (no indentation).

    # Examples
    Article:
    Mario Kart 7 is a 2011 kart racing video game developed by Nintendo EAD in cooperation with Retro Studios and published by Nintendo for the Nintendo 3DS. As with the previous games in the Mario Kart series, players participate in racing on various Mario-themed tracks, playing as one of seventeen different Mario characters. While racing, the players make use of power-up items that either assist their character or hinder opposing characters. New additions to the game include hang-gliding attachments for karts, the ability to drive underwater, the ability to drive in first person, and the ability to fully customize the vehicles' builds. The game supports both local and online multiplayer for up to eight players. The game was a critical and commercial success, becoming the best-selling game on the Nintendo 3DS with over 18.98 million copies sold worldwide.

    Output:
    Example of output:
    {"entities":[{"name":"Mario Kart 7","type":"Game"},{"name":"Nintendo EAD","type":"Developer"},{"name":"Retro Studios","type":"Developer"},{"name":"Nintendo","type":"Publisher"},{"name":"Nintendo 3DS","type":"Console"}],"triples":[["Mario Kart 7","release year","2011"],["Mario Kart 7","genre","Kart Racing"],["Mario Kart 7","developer","Nintendo EAD"],["Mario Kart 7","codeveloper","Retro Studios"],["Mario Kart 7","publisher","Nintendo"],["Mario Kart 7","platform","Nintendo 3DS"],["Mario Kart 7","series","Mario Kart"],["Mario Kart 7","gameplay feature","racing on Mario-themed tracks"],["Mario Kart 7","gameplay feature","power-up items"],["Mario Kart 7","gameplay feature","hang-gliding attachments"],["Mario Kart 7","gameplay feature","underwater driving"],["Mario Kart 7","gameplay feature","first-person driving"],["Mario Kart 7","gameplay feature","vehicle customization"],["Mario Kart 7","supports multiplayer mode","local multiplayer"],["Mario Kart 7","supports multiplayer mode","online multiplayer"],["Mario Kart 7","multiplayer capacity","up to eight players"],["Mario Kart 7","is best selling game on","Nintendo 3DS"],["Mario Kart 7","copies sold worldwide","over 18.98 million"],["Mario Kart 7","reception","critical success"],["Mario Kart 7","reception","commercial success"],["Mario Kart 7","number of playable characters","17"],["Mario Kart 7","features characters from","Mario franchise"]]}

    Article:
    # Final Fantasy Adventure

    Final Fantasy Adventure, known in Japan as Seiken Densetsu: Final Fantasy Gaiden or simply Seiken Densetsu, and later released in Europe as Mystic Quest, is a 1991 action role-playing game developed and published by Square for the Game Boy. It is a spin-off of the Final Fantasy series and the first game in the Mana series. 
    Originally developed under the name Gemma Knights, it features gameplay roughly similar to that of the original The Legend of Zelda, but with the addition of role-playing statistical elements. A remake, Sword of Mana, was released for the Game Boy Advance in 2003, changing the plot and many gameplay aspects. A second remake was released on mobile phones in Japan which improved the graphics and music of the original version. A third remake, Adventures of Mana, was released for iOS, Android, and PlayStation Vita on February 4, 2016.
    The story follows the hero and the heroine as they attempt to thwart the Dark Lord of Glaive and his sorcerer assistant, Julius, from destroying the Tree of Mana and dooming their world. The game was released with many familiar elements of the Final Fantasy series, such as chocobos, but these were later changed to feature common enemies and the gameplay style of the Mana series.
    Final Fantasy Adventure was met with generally positive reviews at the time of its release. Over the course of time, reviewers have considered it one of the best action adventure games on the Game Boy. The game also spawned an entirely new game series, called the Mana series, which became a successful video game role-playing franchise. Its sequel, Secret of Mana, was released in 1993.

    Output:
    {"entities":[{"name":"Final Fantasy Adventure","type":"Game"},{"name":"Square","type":"Publisher"},{"name":"Game Boy","type":"Console"},{"name":"The Legend of Zelda","type":"Game"},{"name":"Sword of Mana","type":"Game"},{"name":"Game Boy Advance","type":"Console"},{"name":"Unnamed Japanese Mobile Remake","type":"Game"},{"name":"Adventures of Mana","type":"Game"},{"name":"PlayStation Vita","type":"Console"},{"name":"Secret of Mana","type":"Game"}],"triples":[["Final Fantasy Adventure","known in Japan as","Seiken Densetsu: Final Fantasy Gaiden"],["Final Fantasy Adventure","known in Japan as","Seiken Densetsu"],["Final Fantasy Adventure","European title","Mystic Quest"],["Final Fantasy Adventure","release year","1991"],["Final Fantasy Adventure","genre","action role-playing game"],["Final Fantasy Adventure","developer","Square"],["Final Fantasy Adventure","publisher","Square"],["Final Fantasy Adventure","platform","Game Boy"],["Final Fantasy Adventure","spin-off of","Final Fantasy series"],["Final Fantasy Adventure","is first game in","Mana series"],["Final Fantasy Adventure","original development name","Gemma Knights"],["Final Fantasy Adventure","gameplay similar to","The Legend of Zelda"],["Final Fantasy Adventure","gameplay feature","role-playing statistical elements"],["Sword of Mana","is remake of","Final Fantasy Adventure"],["Sword of Mana","platform","Game Boy Advance"],["Sword of Mana","release year","2003"],["Sword of Mana","changed plot from","Final Fantasy Adventure"],["Sword of Mana","changed gameplay aspects from","Final Fantasy Adventure"],["Unnamed Japanese Mobile Remake","is remake of","Final Fantasy Adventure"],["Unnamed Japanese Mobile Remake","platform","mobile phones"],["Unnamed Japanese Mobile Remake","release region","Japan"],["Unnamed Japanese Mobile Remake","improved graphics of","Final Fantasy Adventure"],["Unnamed Japanese Mobile Remake","improved music of","Final Fantasy Adventure"],["Adventures of Mana","is remake of","Final Fantasy Adventure"],["Adventures of Mana","platform","iOS"],["Adventures of Mana","platform","Android"],["Adventures of Mana","platform","PlayStation Vita"],["Adventures of Mana","release date","February 4, 2016"],["Adventures of Mana","release year","2016"],["Final Fantasy Adventure","story features","hero and heroine"],["Final Fantasy Adventure","antagonist","Dark Lord of Glaive"],["Final Fantasy Adventure","antagonist","Julius"],["Dark Lord of Glaive","assistant is","Julius"],["Final Fantasy Adventure","story objective","prevent destruction of Tree of Mana"],["Final Fantasy Adventure","features elements from","Final Fantasy series"],["Final Fantasy Adventure","features","chocobos"],["Final Fantasy Adventure","later incorporated elements from","common enemies of Mana series"],["Final Fantasy Adventure","later incorporated elements from","gameplay style of Mana series"],["Final Fantasy Adventure","reception","generally positive reviews at release"],["Final Fantasy Adventure","rated by reviewers as","one of the best action adventure games on Game Boy"],["Final Fantasy Adventure","spawned series","Mana series"],["Mana series","is","successful video game role-playing franchise"],["Secret of Mana","is sequel to","Final Fantasy Adventure"],["Secret of Mana","release year","1993"]]}

    # Input
    Article:
    {{ article }}

  "#
}
test metadata_extraction {
  functions [ExtractArticleMetadata]
  args {
    article #"
      Mario Kart 7 is a 2011 kart racing video game developed by Nintendo EAD in cooperation with Retro Studios and published by Nintendo for the Nintendo 3DS. As with the previous games in the Mario Kart series, players participate in racing on various Mario-themed tracks, playing as one of seventeen different Mario characters. While racing, the players make use of power-up items that either assist their character or hinder opposing characters. New additions to the game include hang-gliding attachments for karts, the ability to drive underwater, the ability to drive in first person, and the ability to fully customize the vehicles' builds. The game supports both local and online multiplayer for up to eight players. The game was a critical and commercial success, becoming the best-selling game on the Nintendo 3DS with over 18.98 million copies sold worldwide.
    "#
  }
}
