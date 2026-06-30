import argparse
import json


DEFAULT_LANGUAGES = ["en", "zh", "fr", "ja", "ko", "es", "pt", "ar"]
DOCS_BUCKET = "docs"


def normalize_doc_value(value):
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    text = str(value)
    return [text] if text.strip() else []


def infer_languages(data, requested_languages):
    if requested_languages:
        return requested_languages
    if isinstance(data, dict):
        langs = [lang for lang in DEFAULT_LANGUAGES if lang in data]
        if langs:
            return langs
    if isinstance(data, list) and data:
        first = data[0]
        if not isinstance(first, dict):
            return []
        first_doc = first.get("doc", {})
        if isinstance(first_doc, dict):
            langs = [lang for lang in DEFAULT_LANGUAGES if lang in first_doc]
            if langs:
                return langs
        if "doc" in first or "docs" in first:
            return [DOCS_BUCKET]
    return []


def load_instances_by_language(data, languages):
    if isinstance(data, dict) and all(lang in data for lang in languages):
        if not all(isinstance(data[lang], list) and data[lang] and isinstance(data[lang][0], dict) for lang in languages):
            raise ValueError("Language-keyed input must contain lists of document instances.")
        return [
            {lang: data[lang][idx] for lang in languages}
            for idx in range(min(len(data[lang]) for lang in languages))
        ]
    if isinstance(data, list):
        return data
    raise ValueError("Unsupported input JSON structure.")


def merge_documents(instances, languages):
    docs_by_lang = {lang: [] for lang in languages}
    for instance in instances:
        for lang in languages:
            if not isinstance(instance, dict):
                continue
            if lang == DOCS_BUCKET and "docs" in instance:
                doc_value = instance["docs"]
            elif lang == DOCS_BUCKET and "doc" in instance:
                doc_value = instance["doc"]
            elif "doc" in instance:
                doc_value = instance["doc"].get(lang) if isinstance(instance["doc"], dict) else instance["doc"]
            else:
                doc_value = instance.get(lang, {}).get("doc")
            docs_by_lang[lang] += normalize_doc_value(doc_value)
    return docs_by_lang


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="input JSON file")
    parser.add_argument("--output", required=True, help="output JSON file")
    parser.add_argument("--languages", nargs="+", default=None, help="languages to merge")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf8") as f:
        data = json.load(f)

    languages = infer_languages(data, args.languages)
    if not languages:
        raise ValueError("No document fields found. Expected language-keyed data, doc, or docs.")
    instances = load_instances_by_language(data, languages)
    docs_by_lang = merge_documents(instances, languages)
    if not any(docs_by_lang.values()):
        raise ValueError("No documents were merged from the input file.")

    gather_docs_all = []
    for lang in languages:
        gather_docs_all += docs_by_lang[lang]

    output = {
        "languages": languages,
        "docs_by_lang": docs_by_lang,
        "gather_docs_all": gather_docs_all,
        "gather_docs_str": " ".join(gather_docs_all),
    }

    with open(args.output, "w", encoding="utf8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
