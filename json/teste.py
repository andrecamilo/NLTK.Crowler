import json 

#modulo de terceiro
import todo

data = 'teste.json'

json.data = json.loads(data)

todos = todo.Todos(data).add('nova ação')

