import pandas as pd


def read_raw_dataset(file_path):
    df = pd.read_json(file_path, lines=True)
    df = df.dropna(subset=['content'])

    df['article'] = df['content'].apply(format_article)
    df = df[df.article.str.contains('\n')] # articles with only a title
    df = df[~df.article.str.endswith(":")] # articles identified as having no information

    df = df.drop_duplicates("article", keep='first')

    return df


def format_article(content):
    article = ""
    if 'title' in content:
        article += f"# {content['title']}\n\n"

    if "summary" in content and len(content["summary"]):
        article += content["summary"]

    if "sections" in content:
        for section in content["sections"]:
            article += format_section(section) + "\n\n"

    return article.strip()


def format_section(section, prefix=""):
    section = ""

    if "title" in section:
        section += f"{prefix}## {section["title"]}\n\n"

    if "text" in section:
        section += f"{section["text"]}\n\n"

    if "subsections" in section:
        section += format_section(section, prefix=prefix+'#')

    return section