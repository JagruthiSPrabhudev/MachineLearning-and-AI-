def buildConnectionString(params):
    return ";".join(["%s=%s" % (k, v) for k, v in params.items()])

if __name__ == "__main__":
    myParams = {"server": "mpiligrim",
                 "database": "master",
                 "uid":"sa",
                 "password": "secret"}
    print(buildConnectionString(myParams))