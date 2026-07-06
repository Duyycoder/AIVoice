import ast
with open("src/main.py", "r", encoding="utf-8") as f:
    tree = ast.parse(f.read())
functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
print(functions)
