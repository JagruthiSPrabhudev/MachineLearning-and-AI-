import sys


inputfile = open(sys.argv[2], "r")

agent ={}
out = open('output.txt', 'w')
for i in inputfile:
    d = i.split(",")
    print(d)
    try:
        location = d[0]
        status = d[1]
        status = status.replace("\n", "")
        print(location)
        print(status)

        if status == "Dirty":
            out.write("Suck"+"\n")
        elif location == "A" and status == "Clean":
            out.write("Right"+"\n")
        elif location == "B" and status == "Clean":
             out.write("Left"+"\n")
    except:
        pass
