import folium

def color_producer(elevation):
    return "green"

map = folium.Map(location=[12.98,80.24],zoom_start=10,tiles="MapBox Bright")
fgv = folium.FeatureGroup(name='Volcanoes')
#fg.add_child(folium.Marker(location=[12.98,80.24],popup="Hello",icon=folium.Icon(color=color_producer(1000))))
fgv.add_child(folium.CircleMarker(location=[12.98,80.24], radius=6, popup="Hello",fill_color=color_producer(1000), color='grey',fill_opacity=0.7))

fgp = folium.FeatureGroup(name='Population')

fgp.add_child(folium.GeoJson(data=open("./input/world.json",'r',encoding='utf-8-sig').read(),
                            style_function=lambda x: {'fillColor':'red' if x['properties']['POP2005'] < 10000000 else 'yellow'}))

map.add_child(fgv)
map.add_child(fgp)
map.add_child(folium.LayerControl())
map.save("Map1.html")
#print(dir(folium))
