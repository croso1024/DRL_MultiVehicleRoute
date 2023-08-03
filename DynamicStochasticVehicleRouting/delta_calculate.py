STC_pair = [
    (2,0.2,12,6),(4,0.15,6,6) , (6,0.1,4,6),
    (2,0.2,6,18),(4,0.15,3,18) , (6,0.1,2,18)
]
ctotal = 70


for S,T,C_sample,cfinal in STC_pair: 
    v = 0 
    delta = 0 
    for i in range(1,S+1): 
        v += (i/(S+1))  *C_sample
    delta = 1/ctotal * (v+cfinal)
    print(f"-- S={S} T={T} C-sample={C_sample} --> delta = {delta}\n")
    