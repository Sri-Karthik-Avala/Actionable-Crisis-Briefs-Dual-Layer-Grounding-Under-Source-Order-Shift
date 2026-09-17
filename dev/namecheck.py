import ast, builtins
src=open("solution.py",encoding="utf-8").read()
tree=ast.parse(src)
defined=set(dir(builtins))
for node in ast.walk(tree):
    if isinstance(node,(ast.FunctionDef,ast.ClassDef)): defined.add(node.name)
    elif isinstance(node,ast.arg): defined.add(node.arg)
    elif isinstance(node,(ast.Import,ast.ImportFrom)):
        for a in node.names: defined.add((a.asname or a.name).split(".")[0])
    elif isinstance(node,ast.Name) and isinstance(node.ctx,(ast.Store,ast.Del)): defined.add(node.id)
used={n.id for n in ast.walk(tree) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Load)}
undef=sorted(used-defined)
assigned_consts={n.id for n in ast.walk(tree) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Store) and n.id.isupper()}
unused_consts=sorted(assigned_consts-used)
print("undefined names:",undef)
print("unused constants:",unused_consts)
