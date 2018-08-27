import sys


inputfile = open(sys.argv[2], "r")

model ={'A':[None],'B': [None]}
out = open('output.txt', 'w')
for i in inputfile:
    d = i.split(",")
    print(d)
    try:
        location = d[0]
        status = d[1]
        status = status.replace("\n", "")
        #print(location)
        #print(status)
        if location in model.keys():
            model[location] = status
            print(model[location])
        if model[location] == "Dirty":
            out.write("Suck"+"\n")
        elif model["A"] == model["B"] == "Clean":
             out.write("NoOp \n")
        elif location == "A":
            out.write("Right"+"\n")
        elif location == "B":
             out.write("Left"+"\n")

        else:
            out.write("WrongInput")
    except:
        pass
out.close()