import json 

a = {"We":12 , "WTW":22 }

with open("./test.json" , "w" ) as file : 
    json.dump(a , file)
