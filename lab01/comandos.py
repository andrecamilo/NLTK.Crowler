# # # # # # # # # 
# lab01 # # # # # 
# # # # # # # # # 
conteudo = open("avaliacoes.json").read()
import json 
avaliacoes = json.loads(conteudo)
avaliacoes 
from similares import inverte
print(json.dumps(inverte(avaliacoes), indent=3))

from similares import sim_euclidiana
sim_euclidiana(avaliacoes, "Rubens V. Martins", "Denis E. Barreto")

from similares import top_similares
top_similares(avaliacoes,  "Rubens V. Martins", 2, sim_euclidiana)
top_similares(inverte(avaliacoes), "Scarface", 2, sim_euclidiana)

from similares import sim_manhattan
sim_manhattan(avaliacoes, "Adriano K. Lopes", "Denis E. Barreto")
top_similares(avaliacoes, "Adriano K. Lopes", 2, sim_manhattan)

from similares import sim_pearson
sim_pearson(inverte(avaliacoes), "A Princesa e o Plebeu", "Os Bons Companheiros")
top_similares(inverte(avaliacoes), "A Princesa e o Plebeu", 2, sim_pearson)

# # # # # # # # # 
# lab02 # # # # # 
# # # # # # # # # 