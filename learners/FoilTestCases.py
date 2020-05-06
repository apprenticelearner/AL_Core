import FoilClassifier as fc 
import numpy as np

smallData = [{("c1"):'0', ("c2"):'0'},
             {("c1"):'1', ("c2"):'0'},
             {("c1"):'0', ("c2"):'1'},
             {("c1"):'1', ("c2"):'1'}]

smallLabel = [0,0,0,2]


smallData2 = np.array([
#    0 1 2 3 4 5 6 7 8 9 10111213141516
    [0,0,1,0,1,1,1,1,1,1,1,0,0,1,1,1,1], #3
    [0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0], #1
    [0,0,0,0,1,0,1,1,1,1,1,0,0,0,0,0,0], #1
    [0,0,1,0,1,0,1,1,1,1,1,0,0,0,0,0,0], #1
    [1,0,1,0,1,0,1,1,1,1,1,0,0,0,0,0,1], #2
    [0,0,1,0,1,1,1,1,1,1,1,0,0,0,0,1,0], #2
    [1,0,1,0,1,0,1,1,1,1,1,0,0,0,0,1,0], #2
    ])

smallLabel2 = np.array([3,1,1,1,2,2,2])


dbx = np.array([[1,0],
               [0,1],
               [1,0],
               [0,1],
               [1,1],
               [1,1],
               [0,0],
               [0,0]])

dby = np.array([1,1,0,0,0,0,1,0])





if(__name__ == "__main__"):
	# When data is a dict list
    # Fit
    cl = fc.FOILClassifier()
    X, dictVect = fc.convertX(smallData, "train")
    cl.fit(X, smallLabel, dictVect)

    # Predict
    Xc, dv = fc.convertX(smallData, "pred", cl.dictVect)
    print("predict:", cl.predictAll(Xc, fc.param.K))

    acc = cl.getRulesAccu()
    print("rules and their accuracy:", acc)


    # When data is an array
    # Fit
    data = fc.addFeature(smallData2)
    cl2 = fc.FOILClassifier(data, smallLabel2)
    cl2.startFOIL()

    # Predict
    pred = cl2.predictAll(data, fc.param.K)
    print("predict:", pred)

    acc2 = cl2.getRulesAccu()
    print("rules and their accuracy:", acc2)


    # made up some data
    data2 = fc.addFeature(dbx)
    cl3 = fc.FOILClassifier(data2, dby)
    cl3.startFOIL()

    # Predict
    pred2 = cl3.predictAll(data2, fc.param.K)
    print("predict:", pred2)

    acc3 = cl3.getRulesAccu()
    print("rules and their accuracy:", acc3)







