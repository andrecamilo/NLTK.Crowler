# -*- coding: utf-8 -*-
# from autoriza_usuario import api

# local = input('Informe a sigla do país desejado(ex. br): ')

# places = api.geo_search(query=local, granularity='country')
# print('O place_id de ' + local +' é: ',places[0].id)


from autoriza_usuario import api
places = api.geo_search(query='br', granularity='country')
print('O place_id de ' + local +' é: ',places[0].id)
