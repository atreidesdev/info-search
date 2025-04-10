import re
from invert_index import load_inverted_index

def boolean_search(query, inverted_index):
    def eval_expr(expr):
        expr = expr.strip()
        if expr.startswith("NOT "):
            term = expr[4:]
            return all_docs - inverted_index.get(term, set())
        else:
            return inverted_index.get(expr, set())

    def tokenize_query(query):
        return re.findall(r'\(|\)|AND|OR|NOT|[^\s()]+', query)

    def parse(tokens):
        def parse_term():
            token = tokens.pop(0)
            if token == "(":
                result = parse_expr()
                tokens.pop(0)  # Remove ')'
                return result
            elif token == "NOT":
                return all_docs - parse_term()
            else:
                return inverted_index.get(token, set())

        def parse_expr():
            result = parse_term()
            while tokens and tokens[0] in ("AND", "OR"):
                op = tokens.pop(0)
                right = parse_term()
                if op == "AND":
                    result = result & right
                elif op == "OR":
                    result = result | right
            return result

        return parse_expr()

    tokens = tokenize_query(query)
    all_docs = set().union(*inverted_index.values())
    return parse(tokens)


if __name__ == "__main__":
    index = load_inverted_index()
    while True:
        user_query = input("Введите запрос: ")
        if not user_query.strip():
            break
        result = boolean_search(user_query, index)
        print("Результаты:", sorted(result))
